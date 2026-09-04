"""
Neural network architectures used in MoPred.
"""

from .voxelmorph import Voxelmorph
from .spatial_transform import SpatialTransformer
from .attention import SelfAttention
from .EMA import EMA

__all__ = [
    "Voxelmorph",
    "SpatialTransformer",
    "SelfAttention",
    "EMA",
]


try:
    from .mambamorph import MambaMorph
    __all__.append("MambaMorph")
except ImportError:
    MambaMorph = None


from .CLDM.UNet3D import UNet3D
from .Context_Encoder.TM_Net import TMNet
from .Context_Encoder.RV_Net import RVNet
from .VAE.DVFVAE import DVFVAE


__all__ += [
    "UNet3D",
    "TMNet",
    "RVNet",
    "DVFVAE",
]