"""Variant registry. Importing this package triggers @register_config + @register
side effects in the variant modules.
"""
from . import mean_diff  # noqa: F401
from . import pca  # noqa: F401
from . import directional_ablation  # noqa: F401
