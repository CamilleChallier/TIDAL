"""
Abstract base class for TIDAL diffusion models.
Defines the shared training loop (add noise, denoise, compute loss), DDIM sampling,
and checkpoint I/O. Concrete models such as UNet3D inherit from ``BaseDiffusionModel``.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noise_schedule import NoiseSchedule
from .outputs import DiffusionOutput
from .dpm_solver_pytorch import NoiseScheduleVP, DPM_Solver
from .dpm_solver_v3 import DPM_Solver_v3
from .dpm_solver_v3 import NoiseScheduleVP as NoiseScheduleVP_v3
from .uni_pc import UniPC
from .uni_pc import NoiseScheduleVP as NoiseScheduleVP_unipc

from ....utils.losses import ncc_loss, gradient_loss


class BaseDiffusionModel(ABC, nn.Module):
    """
    Common diffusion loop shared by LatentDiffusion and CausalDiT.

    Subclasses must implement:
      - encode_context(...)  → context object (model-specific)
      - predict_eps(z_noisy, tau, context) → eps_pred
      - decode_latent(z0) → dvf
      - warp_volume(ref, dvf) → vol
      - encode_dvf(dvf) 

    Subclasses may override:
      - latent_shape  (property)
      - training_forward(...)
      - inference_forward(...)
    """

    schedule: NoiseSchedule  # must be set by subclass __init__

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def encode_context(self, *args, **kwargs):
        """Returns a context object passed to predict_eps."""
        ...

    @abstractmethod
    def predict_eps(
        self,
        z_noisy:  torch.Tensor,
        tau:      torch.Tensor,
        context,
    ) -> torch.Tensor:
        """ε_θ(z_τ, τ, c) → predicted noise."""
        ...

    @abstractmethod
    def encode_dvf(
        self, dvf: torch.Tensor, cond_feats: torch.Tensor = None
    ) -> torch.Tensor:
        """DVF → latent z via the VAE encoder."""
        ...

    @abstractmethod
    def decode_latent(self, z0: torch.Tensor, cond_feats: torch.Tensor = None) -> torch.Tensor:
        """z₀ → DVF via the VAE decoder."""
        ...

    @abstractmethod
    def warp_volume(
        self, ref: torch.Tensor, dvf: torch.Tensor
    ) -> torch.Tensor:
        """Apply spatial transform: ref warped by dvf."""
        ...

    @property
    @abstractmethod
    def latent_shape(self) -> Tuple[int, ...]:
        """Shape of one latent sample (excluding batch dim)."""
        ...

    # ------------------------------------------------------------------
    # Criterion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _call_criterion(criterion, pred, target, device):
        """Call criterion regardless of whether it accepts a `device` kwarg.

        ncc_loss / ncc_as_loss require device to allocate a filter tensor.
        nn.MSELoss and other nn.Module losses do not accept it.
        """
        try:
            return criterion(pred, target, device=device)
        except TypeError:
            return criterion(pred, target)

    # ------------------------------------------------------------------
    # Latent normalisation  (shared by all subclasses)
    # ------------------------------------------------------------------

    def _init_latent_stats(self) -> None:
        """Call once in subclass __init__ to register normalisation buffers."""
        self.register_buffer("latent_mean", torch.zeros(1))
        self.register_buffer("latent_std",  torch.ones(1))
        # Per-phi bin stats (optional — populated by set_latent_stats if available)
        self._phi_bin_means: list = []
        self._phi_bin_stds:  list = []
        self._n_phi_bins:    int  = 0

    def set_latent_stats(self, mean: float, std: float,
                         bin_means: list = None, bin_stds: list = None,
                         n_bins: int = 0) -> None:
        """Set normalisation statistics computed from the training set."""
        self.latent_mean.fill_(mean)
        self.latent_std.fill_(max(std, 1e-6))
        if bin_means and bin_stds:
            self._phi_bin_means = [float(m) for m in bin_means]
            self._phi_bin_stds  = [max(float(s), 1e-6) for s in bin_stds]
            self._n_phi_bins    = n_bins or len(bin_means)
            print(f"[{type(self).__name__}] Latent normalisation: mean={mean:.4f}  std={std:.4f}"
                  f"  + {self._n_phi_bins} per-phi bins loaded")
        else:
            print(f"[{type(self).__name__}] Latent normalisation: mean={mean:.4f}  std={std:.4f}")

    def _phi_stats(self, phi: float):
        """Return (mean, std) for a given phi value using per-bin stats if available."""
        if self._n_phi_bins and self._phi_bin_means:
            bn = min(int(phi / (0.5 / self._n_phi_bins)), self._n_phi_bins - 1)
            return self._phi_bin_means[bn], self._phi_bin_stds[bn]
        return self.latent_mean.item(), self.latent_std.item()

    def _norm(self, z: torch.Tensor, phi: float = None) -> torch.Tensor:
        if phi is not None and self._n_phi_bins:
            m, s = self._phi_stats(phi)
            return (z - m) / s
        return (z - self.latent_mean) / self.latent_std

    def _denorm(self, z: torch.Tensor, phi: float = None) -> torch.Tensor:
        if phi is not None and self._n_phi_bins:
            m, s = self._phi_stats(phi)
            return z * s + m
        return z * self.latent_std + self.latent_mean

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        return mu + torch.randn_like(mu) * (0.5 * log_var).exp()

    def _kl_loss(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    def _tweedie_z0(
        self,
        z_tau:    torch.Tensor,
        tau:      torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate z₀ from z_τ and predicted noise via the Tweedie formula."""
        sqrt_ab  = self.schedule.sqrt_alpha_bar[tau].view(-1, 1, 1, 1, 1)
        sqrt_1ab = self.schedule.sqrt_one_minus_alpha_bar[tau].view(-1, 1, 1, 1, 1)
        return (z_tau - sqrt_1ab * eps_pred) / sqrt_ab

    # ------------------------------------------------------------------
    # CFG helpers
    # ------------------------------------------------------------------

    def _make_null_context(self, c: torch.Tensor) -> torch.Tensor:
        """Return the null (unconditional) context token used during CFG inference."""
        return torch.zeros_like(c)

    # ------------------------------------------------------------------
    # Self-Swap Guidance (SSG) helpers — Zhang et al., "Guiding a Diffusion
    # Model by Swapping Its Tokens", CVPR 2026.
    # ------------------------------------------------------------------

    def _set_self_swap(self, enabled: bool, ratio: float = 0.0, mode: str = "both") -> None:
        """
        Toggle the token-swap perturbation used by ssg_sample.

        No-op by default — override in subclasses whose denoiser exposes
        token-bearing blocks (e.g. self-attention) that support swapping.
        """
        return

    # ------------------------------------------------------------------
    # Shared DDIM sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        context,
        batch_size:   int,
        device:       torch.device,
        ddim_steps:   int   = 50,
        cfg_scale:    float = 1.0,
        predict_mode: str   = "eps",
        z_init:       "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """
        Reverse DDIM loop (eta=0) with optional CFG and v-prediction support.

        predict_mode="eps": network predicts ε (standard).
        predict_mode="v":   network predicts v; converted to ε before each DDIM step.

        CFG combination is performed in the network's native prediction space
        (eps or v), then converted to eps for the DDIM update rule.

        z_init: optional fixed starting noise. Passing the same tensor across
        all test samples removes per-sample noise jitter so that temporal
        evolution is driven purely by the conditioning signal.
        """
        z         = z_init if z_init is not None else torch.randn(batch_size, *self.latent_shape, device=device)
        step_size = self.schedule.T // ddim_steps
        timesteps = list(range(self.schedule.T, 0, -step_size))
        null_ctx  = self._make_null_context(context) if cfg_scale > 1.0 else None

        for i, tau_idx in enumerate(timesteps):
            tau       = torch.full((batch_size,), tau_idx - 1,
                                   device=device, dtype=torch.long)
            out_cond  = self.predict_eps(z, tau, context)

            if cfg_scale > 1.0:
                out_uncond = self.predict_eps(z, tau, null_ctx)
                out        = out_uncond + cfg_scale * (out_cond - out_uncond)
            else:
                out = out_cond

            # Convert network output to ε for the DDIM update rule
            if predict_mode == "v":
                eps_pred = self.schedule.predict_eps_from_v(z, tau, out)
            else:
                eps_pred = out

            tau_prev_idx = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            if tau_prev_idx == 0:
                z = self._tweedie_z0(z, tau, eps_pred)
            else:
                z = self.schedule.ddim_step(z, tau_idx, tau_prev_idx, eps_pred)

        return z

    @torch.no_grad()
    def _get_compiled_predict_eps(self):
        """Return a torch.compile'd version of predict_eps, built once and cached."""
        if not hasattr(self, "_compiled_predict_eps_fn"):
            object.__setattr__(
                self, "_compiled_predict_eps_fn",
                torch.compile(self.predict_eps, dynamic=True, fullgraph=False),
            )
        return self._compiled_predict_eps_fn

    def _get_dpm_ns(self, device: torch.device) -> "NoiseScheduleVP":
        """Return a cached NoiseScheduleVP with arrays already on `device`.

        Avoids recreating and re-uploading the schedule on every forward pass,
        which was causing repeated CPU→GPU copies inside the DPM-Solver loop.
        """
        cache_key = f"_dpm_ns_{device}"
        if not hasattr(self, cache_key):
            ns = NoiseScheduleVP(schedule='discrete', betas=self.schedule.betas)
            # Pre-place the lookup arrays on the target device so that
            # marginal_log_mean_coeff / inverse_lambda never trigger a transfer.
            ns.t_array         = ns.t_array.to(device)
            ns.log_alpha_array = ns.log_alpha_array.to(device)
            object.__setattr__(self, cache_key, ns)
        return getattr(self, cache_key)

    def dpm_solver_sample(
        self,
        context,
        batch_size:   int,
        device:       torch.device,
        dpm_steps:    int   = 20,
        order:        int   = 2,
        cfg_scale:    float = 1.0,
        predict_mode: str   = "eps",
        z_init:       "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """
        DPM-Solver++ reverse sampler (Lu et al. 2022).

        Reaches comparable quality to DDIM in 2-5× fewer NFE by using a
        higher-order ODE solver instead of first-order Euler steps.

        order=2 (multistep) is the recommended default: ~20 steps match
        ~50 DDIM steps.  order=3 can converge in ~15 steps.
        """
        T           = self.schedule.T
        ns          = self._get_dpm_ns(device)
        predict_eps = self._get_compiled_predict_eps()
        null_ctx    = self._make_null_context(context) if cfg_scale > 1.0 else None

        def model_fn(x, t_continuous):
            # DPM-Solver passes a scalar t per batch element; convert to discrete index.
            tau = (t_continuous * T - 1).long().clamp(0, T - 1)
            out = predict_eps(x, tau, context)
            if cfg_scale > 1.0:
                out_u = predict_eps(x, tau, null_ctx)
                out   = out_u + cfg_scale * (out - out_u)
            if predict_mode == "v":
                out = self.schedule.predict_eps_from_v(x, tau, out)
            return out

        solver = DPM_Solver(model_fn, ns, algorithm_type="dpmsolver++")
        z = z_init if z_init is not None else torch.randn(batch_size, *self.latent_shape, device=device)
        return solver.sample(z, steps=dpm_steps, order=order,
                             skip_type='time_uniform', method='multistep')

    def _get_dpm_v3_solver(self, device: torch.device, dpm_steps: int,
                           statistics_dir: str | None, order: int) -> "DPM_Solver_v3":
        """Return a cached DPM_Solver_v3 for a given (steps, stats_dir) pair.

        DPM_Solver_v3.__init__ precomputes the timestep grid and exponential
        integral tables, so caching avoids repeating this work every forward pass.
        """
        cache_key = f"_dpmv3_{dpm_steps}_{statistics_dir}"
        if not hasattr(self, cache_key):
            ns = NoiseScheduleVP_v3(schedule='discrete', betas=self.schedule.betas.cpu())
            ns.t_array = ns.t_array.to(device)
            ns.log_alpha_array = ns.log_alpha_array.to(device)
            degenerated = statistics_dir is None
            solver = DPM_Solver_v3(
                statistics_dir=statistics_dir,
                noise_schedule=ns,
                steps=dpm_steps,
                skip_type='time_uniform',
                degenerated=degenerated,
                device=str(device),
            )
            object.__setattr__(self, cache_key, solver)
        return getattr(self, cache_key)

    def dpm_solver_v3_sample(
        self,
        context,
        batch_size:     int,
        device:         torch.device,
        dpm_steps:      int   = 20,
        order:          int   = 2,
        statistics_dir: str | None = None,
        cfg_scale:      float = 1.0,
        predict_mode:   str   = "eps",
        z_init:         "torch.Tensor | None" = None,
        use_corrector:  bool  = False,
        p_pseudo:       bool  = False,
        c_pseudo:       bool  = False,
    ) -> torch.Tensor:
        """
        DPM-Solver v3 sampler (Zheng et al. 2023).

        Uses precomputed EMS statistics for optimal exponential integrators.
        When statistics_dir is None, runs in degenerated mode (equivalent to
        a first-order solver without data-adaptive coefficients) — useful for
        speed testing before statistics are computed.

        statistics_dir : path containing l.npz and sb.npz generated by
                         compute_EMS_scoresde.py.  None = degenerated mode.
        order          : predictor order (1, 2, or 3).
        use_corrector  : enable PC correction step (doubles NFE).
        """
        T           = self.schedule.T
        predict_eps = self._get_compiled_predict_eps()
        null_ctx    = self._make_null_context(context) if cfg_scale > 1.0 else None

        def model_fn(x, t_continuous):
            tau = (t_continuous * T - 1).long().clamp(0, T - 1)
            out = predict_eps(x, tau, context)
            if cfg_scale > 1.0:
                out_u = predict_eps(x, tau, null_ctx)
                out   = out_u + cfg_scale * (out - out_u)
            if predict_mode == "v":
                out = self.schedule.predict_eps_from_v(x, tau, out)
            return out

        solver = self._get_dpm_v3_solver(device, dpm_steps, statistics_dir, order)
        z = z_init if z_init is not None else torch.randn(batch_size, *self.latent_shape, device=device)
        return solver.sample(
            z, model_fn,
            order=order,
            p_pseudo=p_pseudo,
            use_corrector=use_corrector,
            c_pseudo=c_pseudo,
            lower_order_final=True,
        )

    def _get_unipc_ns(self, device: torch.device) -> "NoiseScheduleVP_unipc":
        """Cached UniPC NoiseScheduleVP with arrays pre-placed on `device`."""
        cache_key = f"_unipc_ns_{device}"
        if not hasattr(self, cache_key):
            ns = NoiseScheduleVP_unipc(schedule='discrete', betas=self.schedule.betas)
            ns.t_array         = ns.t_array.to(device)
            ns.log_alpha_array = ns.log_alpha_array.to(device)
            object.__setattr__(self, cache_key, ns)
        return getattr(self, cache_key)

    def uni_pc_sample(
        self,
        context,
        batch_size:   int,
        device:       torch.device,
        upc_steps:    int   = 20,
        order:        int   = 2,
        variant:      str   = "bh1",
        cfg_scale:    float = 1.0,
        predict_mode: str   = "eps",
        z_init:       "torch.Tensor | None" = None,
        skip_type:    str   = "time_uniform",
        no_corrector: bool  = False,
    ) -> torch.Tensor:
        """
        UniPC sampler (Zhao et al. 2023) — Unified Predictor-Corrector.

        variant      : 'bh1' (default, noise-prediction-friendly) or 'bh2'.
        order        : 1, 2, or 3.
        skip_type    : 'time_uniform' (default) or 'logSNR' (often better at low steps).
        no_corrector : predictor-only mode — skips corrector arithmetic every step.
        """
        T           = self.schedule.T
        ns          = self._get_unipc_ns(device)
        predict_eps = self.predict_eps
        null_ctx    = self._make_null_context(context) if cfg_scale > 1.0 else None

        def model_fn(x, t_continuous):
            tau = (t_continuous * T - 1).long().clamp(0, T - 1)
            out = predict_eps(x, tau, context)
            if cfg_scale > 1.0:
                out_u = predict_eps(x, tau, null_ctx)
                out   = out_u + cfg_scale * (out - out_u)
            if predict_mode == "v":
                out = self.schedule.predict_eps_from_v(x, tau, out)
            return out

        sampler = UniPC(model_fn, ns, algorithm_type="noise_prediction", variant=variant)
        z = z_init if z_init is not None else torch.randn(batch_size, *self.latent_shape, device=device)
        return sampler.sample(z, steps=upc_steps, order=order,
                              skip_type=skip_type, method='multistep',
                              no_corrector=no_corrector)

    def _sample(
        self,
        sampler:      str,
        context,
        batch_size:   int,
        device:       torch.device,
        steps:        int,
        cfg_scale:    float = 1.0,
        predict_mode: str   = "eps",
        z_init:       "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Dispatch to the requested sampler. Use this in inference paths instead of
        calling ddim_sample / dpm_solver_sample directly."""
        if sampler == "ddim":
            return self.ddim_sample(context, batch_size, device, steps,
                                    cfg_scale=cfg_scale, predict_mode=predict_mode, z_init=z_init)
        elif sampler in ("dpm++", "dpm++2"):
            return self.dpm_solver_sample(context, batch_size, device, steps,
                                          order=2, cfg_scale=cfg_scale,
                                          predict_mode=predict_mode, z_init=z_init)
        elif sampler == "dpm++3":
            return self.dpm_solver_sample(context, batch_size, device, steps,
                                          order=3, cfg_scale=cfg_scale,
                                          predict_mode=predict_mode, z_init=z_init)
        elif sampler.startswith("dpmv3"):
            # "dpmv3"  → order=2, "dpmv3_3" → order=3, "dpmv3_1" → order=1
            parts = sampler.split("_")
            v3_order = int(parts[1]) if len(parts) > 1 else 2
            stats_dir = getattr(self, "dpmv3_stats_dir", None)
            return self.dpm_solver_v3_sample(context, batch_size, device, steps,
                                             order=v3_order,
                                             statistics_dir=stats_dir,
                                             cfg_scale=cfg_scale,
                                             predict_mode=predict_mode,
                                             z_init=z_init)
        elif sampler.startswith("unipc"):
            # Naming: unipc[_<order>][_bh2][_logsnr][_nc]
            # Examples: unipc, unipc_3, unipc_bh2, unipc_logsnr, unipc_nc, unipc_3_logsnr_nc
            parts        = sampler.split("_")
            upc_order    = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 2
            upc_variant  = "bh2" if "bh2" in parts else "bh1"
            upc_skiptype = "logSNR" if "logsnr" in parts else "time_uniform"
            upc_nc       = "nc" in parts
            return self.uni_pc_sample(context, batch_size, device, steps,
                                      order=upc_order, variant=upc_variant,
                                      skip_type=upc_skiptype, no_corrector=upc_nc,
                                      cfg_scale=cfg_scale,
                                      predict_mode=predict_mode,
                                      z_init=z_init)
        else:
            raise ValueError(
                f"Unknown sampler {sampler!r}. "
                "Choose from: ddim, dpm++, dpm++2, dpm++3, dpmv3, dpmv3_1, dpmv3_3, "
                "unipc, unipc_3, unipc_bh2, unipc_logsnr, unipc_nc"
            )

    @torch.no_grad()
    def ssg_sample(
        self,
        context,
        batch_size:   int,
        device:       torch.device,
        ddim_steps:   int   = 50,
        ssg_scale:    float = 1.0,
        swap_ratio:   float = 0.1,
        swap_mode:    str   = "both",
        predict_mode: str   = "eps",
    ) -> torch.Tensor:
        """
        Reverse DDIM loop (eta=0) guided by Self-Swap Guidance (SSG)
        — Zhang et al., "Guiding a Diffusion Model by Swapping Its Tokens"
        (CVPR 2026 Oral) — instead of CFG. Training-free: requires no
        unconditional branch and works with any pretrained checkpoint.

        At each step the denoiser is run twice with shared weights:
          - ε_ori  : clean forward pass (token-swap perturbation disabled)
          - ε_pert : forward pass with the `swap_ratio` fraction of the most
                     semantically-dissimilar token pairs exchanged (by cosine
                     similarity) inside every self-attention block — the
                     paper's "degradation branch"

        and combined via the paper's guidance formula (Eq. for ε̃):
            ε̃(z_τ) = ε_ori(z_τ) + ω · (ε_ori(z_τ) − ε_pert(z_τ))
        where ω = ssg_scale.

        Subclasses must implement _set_self_swap to expose token swapping on
        their denoiser (default is a no-op, i.e. ssg_scale has no effect).
        """
        print(
            f"[SSG] ssg_sample called — steps={ddim_steps}  scale={ssg_scale}"
            f"  swap_ratio={swap_ratio}  mode={swap_mode}  active={ssg_scale > 1.0}"
        )

        z         = torch.randn(batch_size, *self.latent_shape, device=device)
        step_size = self.schedule.T // ddim_steps
        timesteps = list(range(self.schedule.T, 0, -step_size))

        for i, tau_idx in enumerate(timesteps):
            tau = torch.full((batch_size,), tau_idx - 1,
                             device=device, dtype=torch.long)

            self._set_self_swap(False)
            out_ori = self.predict_eps(z, tau, context)

            if ssg_scale > 1.0:
                self._set_self_swap(True, ratio=swap_ratio, mode=swap_mode)
                out_pert = self.predict_eps(z, tau, context)
                self._set_self_swap(False)
                guidance = ssg_scale * (out_ori - out_pert)
                if i == 0:
                    print(
                        f"[SSG] step 0 guidance norm={guidance.norm():.4f}"
                        f"  ori_norm={out_ori.norm():.4f}"
                        f"  pert_norm={out_pert.norm():.4f}"
                    )
                out = out_ori + guidance
            else:
                out = out_ori

            # Convert network output to ε for the DDIM update rule
            if predict_mode == "v":
                eps_pred = self.schedule.predict_eps_from_v(z, tau, out)
            else:
                eps_pred = out

            tau_prev_idx = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            if tau_prev_idx == 0:
                z = self._tweedie_z0(z, tau, eps_pred)
            else:
                z = self.schedule.ddim_step(z, tau_idx, tau_prev_idx, eps_pred)

        return z

    # ------------------------------------------------------------------
    # Shared training step for one horizon timestep
    # ------------------------------------------------------------------
    
    def _diffusion_step(
        self,
        dvf_gt:         torch.Tensor,
        context,
        snr_gamma:      float = 5.0,
        low_tau_frac:   float = 0.25,
        criterion:      callable = ncc_loss,
        cond_feats:     torch.Tensor = None,
        spatial_weight: torch.Tensor | float = 1.0,
        predict_mode:   str   = "eps",
        dose_dropout_p: float = 0.0,
        phi:            float = None,
        recon_weight:   float = 0.0,
        diversity_weight: float = 0.0,
        need_dvf_hat:   bool  = False,

    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward-diffuse one ground-truth DVF and compute:
          - ddpm_loss  (min-SNR-γ weighted MSE on the denoising target)
          - recon_loss (low-τ auxiliary loss, or None)

        predict_mode="eps": target is ε, standard noise prediction.
        predict_mode="v":   target is v = √ᾱ·ε − √(1−ᾱ)·z₀.

        Returns (ddpm_loss, recon_loss, dvf_hat).
        """
        B = dvf_gt.shape[0]

        z0 = self.encode_dvf(dvf_gt, cond_feats, phi=phi)
        if torch.isnan(z0).any() or torch.isinf(z0).any():
            raise RuntimeError(f"NaN/Inf in z0 from VAE encoder! "
                            f"z0 range: [{z0.min():.3f}, {z0.max():.3f}]")
        if z0.abs().max() > 100:
            print(f"[Warning] Large z0 values: max={z0.abs().max():.2f} — clipping")
            z0 = z0.clamp(-10, 10)

        tau        = torch.randint(0, self.schedule.T, (B,), device=dvf_gt.device)
        z_tau, eps = self.schedule.q_sample(z0, tau)
        if dose_dropout_p > 0.0:
            r = torch.bernoulli(
                torch.full((B,), 1.0 - dose_dropout_p, device=dvf_gt.device)
            )  # (B,) — 1=keep, 0=drop
            # print(f"Dose dropout: keeping {r.mean().item() * 100:.1f}% of samples")
            mask  = r.view(-1, 1, 1, 1, 1)
            z_tau = z_tau * mask + torch.randn_like(z_tau) * (1.0 - mask)
        
        net_out = self.predict_eps(z_tau, tau, context)

        # Denoising target — depends on predict_mode
        target = (self.schedule.get_v_target(z0, eps, tau)
                  if predict_mode == "v" else eps)

        # min-SNR-γ weighting (Hang et al. 2023)
        # v-prediction: weight = min(SNR, γ) / (SNR + 1)
        # eps-prediction: weight = min(SNR, γ) / SNR
        snr = self.schedule.alpha_bar[tau] / (1 - self.schedule.alpha_bar[tau])
        if predict_mode == "v":
            snr_weight = (snr.clamp(max=snr_gamma) / (snr + 1)).view(-1, 1, 1, 1, 1)
        else:
            snr_weight = (snr.clamp(max=snr_gamma) / snr).view(-1, 1, 1, 1, 1)
        ddpm_loss  = (snr_weight * spatial_weight * F.mse_loss(net_out, target, reduction="none")).mean()

        # Optional low-τ reconstruction loss.
        # z0_hat is derived from a clean-context prediction so that CFG dropout
        # does not corrupt the Tweedie estimate and degrade the recon gradients.
        recon_loss = None
        dvf_hat    = None
        if low_tau_frac > 0 and (recon_weight > 0 or need_dvf_hat):
            low_mask = tau < int(self.schedule.T * low_tau_frac)
            if low_mask.any():
                z0_hat = (self.schedule.predict_z0_from_v(z_tau, tau, net_out)
                          if predict_mode == "v"
                          else self._tweedie_z0(z_tau, tau, net_out))

                if diversity_weight > 0.0 and B > 1:
                    ddpm_loss = ddpm_loss + diversity_weight * (-z0_hat.std(dim=0).mean())

                dvf_hat = self.decode_latent(z0_hat, cond_feats)

                if recon_weight > 0:
                    ncc_vals = []
                    for ch in range(dvf_hat.shape[1]):
                        ncc_ch = self._call_criterion(
                            criterion,
                            dvf_hat[low_mask, ch:ch+1],
                            dvf_gt[low_mask,  ch:ch+1],
                            dvf_gt.device,
                        )
                        if not (torch.isnan(ncc_ch) or torch.isinf(ncc_ch)):
                            ncc_vals.append(ncc_ch)
                    if ncc_vals:
                        recon_loss = torch.stack(ncc_vals).mean()

        return ddpm_loss, recon_loss, dvf_hat