from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .registry import ModelBuildConfig, register_model


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
DEFAULT_CROP_SIZE_MNIST = 28

PAI_ARG_DEFAULTS = {
    "batch_size": 64,
    "test_batch_size": 1000,
    "epochs": 14,
    "lr": 1.0,
    "gamma": 0.7,
    "weight_decay": 0.0,
    "finetune_backbone": True,
    "num_workers": 4,
}

BASELINE_ARG_DEFAULTS = {
    "batch_size": 64,
    "test_batch_size": 1000,
    "epochs": 14,
    "lr": 1.0,
    "gamma": 0.7,
    "weight_decay": 0.0,
    "finetune_backbone": True,
    "num_workers": 4,
}


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.pre_fc = nn.Linear(9216, 128)
        self.classifier_fc = nn.Linear(128, num_classes)
        self.uses_log_softmax = True

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.pre_fc(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.classifier_fc(x)
        return F.log_softmax(x, dim=1)


SimpleCNNPAI = SimpleCNN


def simple_cnn_mnist(num_classes: int = 10, finetune_backbone: bool = True):
    del finetune_backbone
    return SimpleCNN(num_classes=num_classes)


def wrap_simple_cnn_pai(base_model: nn.Module):
    return base_model


def build_transforms_simple_cnn_mnist():
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    return eval_transform, eval_transform, eval_transform, DEFAULT_CROP_SIZE_MNIST


@register_model("simple_cnn")
def _register_simple_cnn() -> ModelBuildConfig:
    return ModelBuildConfig(
        build_model=simple_cnn_mnist,
        wrap_model=wrap_simple_cnn_pai,
        num_classes=10,
        build_transforms=build_transforms_simple_cnn_mnist,
        pai_convert_targets={"pre_fc": ".pre_fc", "classifier_fc": ".classifier_fc"},
        optimizer_trainable_submodules=("pre_fc", "classifier_fc"),
        pai_arg_defaults=PAI_ARG_DEFAULTS,
        baseline_arg_defaults=BASELINE_ARG_DEFAULTS,
    )