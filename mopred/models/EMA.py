"""
Exponential Moving Average (EMA) wrapper for model parameters.
Maintains a shadow copy of weights with configurable decay and provides
``update``, ``apply_shadow``, and ``restore`` methods for training and evaluation.
"""
import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self._backup: dict = {}

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v

    def apply_shadow(self):
        self._backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self._backup)
        self._backup = {}

    def save(self, path: str):
        torch.save(self.shadow, path)

    def load(self, path: str, device):
        self.shadow = torch.load(path, map_location=device)
        self.model.load_state_dict(self.shadow)