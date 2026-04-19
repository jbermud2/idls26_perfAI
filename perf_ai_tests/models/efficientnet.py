from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b5

try:
    from torchvision.models import EfficientNet_B5_Weights
except ImportError:
    EfficientNet_B5_Weights = None  # type: ignore

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_CLASSES = 102
DEFAULT_CROP_SIZE = 456
DEFAULT_RESIZE_SIZE = 466


def _efficientnet_b5_imagenet_weights():
    """Return EfficientNet-B5 pretrained weights if available."""
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
    weights = _efficientnet_b5_imagenet_weights()

    if weights is not None:
        model = efficientnet_b5(weights=weights)
    else:
        try:
            model = efficientnet_b5(weights="DEFAULT")
        except Exception:
            try:
                model = efficientnet_b5(pretrained=True)  # type: ignore[call-arg]
            except Exception:
                model = efficientnet_b5(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if not finetune_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    return model


class EfficientNetB5PAI(nn.Module):
    """EfficientNet-B5 wrapper with a pre-FC layer for dendrite placement."""

    def __init__(self, efficientnet_model: nn.Module):
        super().__init__()
        self.features = efficientnet_model.features
        self.avgpool = efficientnet_model.avgpool

        fc_in_features = efficientnet_model.classifier[1].in_features
        self.pre_fc = nn.Linear(fc_in_features, fc_in_features)

        self.classifier_dropout = efficientnet_model.classifier[0]
        if hasattr(self.classifier_dropout, "inplace"):
            self.classifier_dropout.inplace = False
        self.classifier_fc = efficientnet_model.classifier[1]

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.pre_fc(x)
        x = F.relu(x, inplace=False)
        x = self.classifier_dropout(x)
        x = self.classifier_fc(x)
        return x


def build_transforms():
    weights = _efficientnet_b5_imagenet_weights()

    crop_size = DEFAULT_CROP_SIZE
    resize_size = DEFAULT_RESIZE_SIZE

    if weights is not None:
        try:
            eval_transform = weights.transforms()
        except Exception:
            eval_transform = transforms.Compose(
                [
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
    else:
        eval_transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform, eval_transform, crop_size
