"""Backward-compatible EfficientNet exports.

Model registrations now live in separate modules:
- models.efficientnet_b4
- models.efficientnet_b5
"""

from .efficientnet_b4 import (
    DEFAULT_CROP_SIZE_B4,
    DEFAULT_RESIZE_SIZE_B4,
    build_transforms_efficientnet_b4,
    efficientnet_b4_flowers102,
)
from .efficientnet_b5 import (
    DEFAULT_CROP_SIZE,
    DEFAULT_RESIZE_SIZE,
    build_transforms_efficientnet_b5,
    efficientnet_b5_flowers102,
)
from .efficientnet_common import (
    EfficientNetB5PAI,
    EfficientNetPAI,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
)


def build_transforms():
    return build_transforms_efficientnet_b5()
