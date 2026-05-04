import os as _os

if _os.environ.get("BEARTYPE"):
    from beartype.claw import beartype_this_package as _bt
    _bt()

from .config import SteeringConfig, REGISTRY, register
from .extract import record_activations
from .attach import attach, detach, save, load, train
from .calibrate import measure_kl, calibrate_iso_kl
from . import variants  # noqa: F401  triggers method + config registration
from .vector import Vector

from .variants.mean_diff import MeanDiffC
from .variants.pca import PCAC
from .variants.directional_ablation import DirectionalAblationC

__all__ = [
    "SteeringConfig",
    "MeanDiffC",
    "PCAC",
    "DirectionalAblationC",
    "record_activations",
    "train",
    "attach",
    "detach",
    "save",
    "load",
    "measure_kl",
    "calibrate_iso_kl",
    "REGISTRY",
    "register",
    "Vector",
]
