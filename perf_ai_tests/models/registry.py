from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple

import torch.nn as nn


@dataclass(frozen=True)
class ModelBuildConfig:
    build_model: Callable[[int, bool], nn.Module]
    """(num_classes, finetune_backbone) -> module passed to perforate_model."""

    wrap_model: Callable[[nn.Module], nn.Module]
    """Wraps the base model with any PAI-specific modules before training."""

    num_classes: int
    """Number of output classes for the configured dataset."""

    build_transforms: Callable[[], Tuple[object, object, object, int]]
    """Returns train_transform, val_transform, test_transform, crop_size."""

    pai_convert_targets: Mapping[str, str]
    """CLI flag value -> dotted module id (e.g. pre_fc -> .pre_fc)."""

    optimizer_trainable_submodules: Tuple[str, ...]
    """Submodule names to force trainable for PAI optimizer setup (when present)."""

    pai_arg_defaults: Mapping[str, object]
    """Default argparse values for main.py (PAI training) keyed by argparse dest names."""

    baseline_arg_defaults: Mapping[str, object]
    """Default argparse values for main_baseline.py keyed by argparse dest names."""


MODEL_BUILD_CONFIGS: Dict[str, ModelBuildConfig] = {}


def register_model(name: str):
    def decorator(build_config: Callable[[], ModelBuildConfig]):
        MODEL_BUILD_CONFIGS[name] = build_config()
        return build_config

    return decorator


def get_model_build_config(name: str) -> ModelBuildConfig:
    if name not in MODEL_BUILD_CONFIGS:
        known = ", ".join(sorted(MODEL_BUILD_CONFIGS.keys())) or "<none>"
        raise KeyError(f"Unknown model '{name}'. Known models: {known}")
    return MODEL_BUILD_CONFIGS[name]


def list_model_build_config_names() -> Tuple[str, ...]:
    return tuple(sorted(MODEL_BUILD_CONFIGS.keys()))


def _import_builtin_model_modules():
    import models.efficientnet_b4 as _efficientnet_b4
    import models.efficientnet_b5 as _efficientnet_b5
    import models.simple_cnn as _simple_cnn


_import_builtin_model_modules()
