from __future__ import annotations

import torch.nn as nn
from torchvision.models import efficientnet_b5

from .efficientnet_common import (
    NUM_CLASSES,
    PAI_CONVERT_TARGETS,
    build_transforms_for_model,
    load_efficientnet,
)
from .registry import ModelBuildConfig, register_model

try:
    from torchvision.models import EfficientNet_B5_Weights
except ImportError:
    EfficientNet_B5_Weights = None  # type: ignore

DEFAULT_CROP_SIZE = 456
DEFAULT_RESIZE_SIZE = 466

PAI_ARG_DEFAULTS = {
    "batch_size": 16,
    "test_batch_size": 128,
    "epochs": 50,
    "lr": 6e-5,
    "weight_decay": 4e-3,
    "finetune_backbone": True,
    "num_workers": 4,
}

BASELINE_ARG_DEFAULTS = {
    "batch_size": 16,
    "test_batch_size": 128,
    "epochs": 50,
    "lr": 6e-5,
    "weight_decay": 4e-3,
    "finetune_backbone": True,
    "num_workers": 4,
}


def _efficientnet_b5_imagenet_weights():
    if EfficientNet_B5_Weights is None:
        return None
    try:
        return EfficientNet_B5_Weights.DEFAULT
    except AttributeError as e:
        print(f"Exception at loading EfficientNet Weights: {e}")
        return EfficientNet_B5_Weights.IMAGENET1K_V1


def efficientnet_b5_flowers102(
    num_classes: int = NUM_CLASSES,
    finetune_backbone: bool = False,
):
    model = load_efficientnet(efficientnet_b5, EfficientNet_B5_Weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if not finetune_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    return model


def build_transforms_efficientnet_b5():
    return build_transforms_for_model(
        _efficientnet_b5_imagenet_weights(),
        DEFAULT_CROP_SIZE,
        DEFAULT_RESIZE_SIZE,
    )


@register_model("efficientnet_b5")
def _register_efficientnet_b5() -> ModelBuildConfig:
    return ModelBuildConfig(
        build_model=efficientnet_b5_flowers102,
        build_transforms=build_transforms_efficientnet_b5,
        pai_convert_targets=PAI_CONVERT_TARGETS,
        optimizer_trainable_submodules=("pre_fc", "classifier_fc"),
        pai_arg_defaults=PAI_ARG_DEFAULTS,
        baseline_arg_defaults=BASELINE_ARG_DEFAULTS,
    )
