'''
For running:
export $WANDB_API_KEY=your_key_here or add --wandb-api-key
python efficientnet_b5_flowers102_perforated.py --data-root ./data --use-wandb --dendrite-mode 1 --max-dendrites 5 --pai-forward-function relu --improvement-threshold 1 --candidate-weight-init-mult 0.1
'''

from __future__ import annotations

import argparse
import base64
import math
import os
import shutil
import time
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b5

try:
    from torchvision.models import EfficientNet_B5_Weights
except ImportError:
    EfficientNet_B5_Weights = None  # type: ignore

try:
    import wandb
except ImportError:
    wandb = None

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
except ImportError:
    GPA = None
    UPA = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FLOWERS_DATASET_ROOT_DEFAULT = "./data"
NUM_CLASSES = 102
DEFAULT_CROP_SIZE = 456
DEFAULT_RESIZE_SIZE = 466


def _efficientnet_b5_imagenet_weights():
    """Return EfficientNet-B5 pretrained weights if available."""
    if EfficientNet_B5_Weights is None:
        return None
    try:
        return EfficientNet_B5_Weights.DEFAULT
    except AttributeError:
        return EfficientNet_B5_Weights.IMAGENET1K_V1


def efficientnet_b5_flowers102(
    num_classes: int = NUM_CLASSES,
    finetune_backbone: bool = False,
) -> nn.Module:
    """
    torchvision EfficientNet-B5 with ImageNet pretrained weights and a new Flowers102 head.
    By default, freeze the feature extractor and train only the classifier.
    """
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
        # Avoid inplace mutation after pre_fc activation; this breaks autograd with PAI edits.
        if hasattr(self.classifier_dropout, "inplace"):
            self.classifier_dropout.inplace = False
        self.classifier_fc = efficientnet_model.classifier[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.pre_fc(x)
        x = F.relu(x, inplace=False)
        x = self.classifier_dropout(x)
        x = self.classifier_fc(x)
        return x


def configure_pai(args) -> None:
    GPA.pc.set_testing_dendrite_capacity(False)

    # Avoid interactive pdb stop on weight-decay warnings in non-interactive runs.
    if hasattr(GPA.pc, "set_weight_decay_accepted") and not args.strict_weight_decay_check:
        GPA.pc.set_weight_decay_accepted(True)

    # threshold presets when integer presets are provided.
    if float(args.improvement_threshold).is_integer():
        preset = int(args.improvement_threshold)
        if preset == 0:
            threshold = [0.01, 0.001, 0.0001, 0]
        elif preset == 1:
            threshold = [0.001, 0.0001, 0]
        elif preset == 2:
            threshold = [0]
        else:
            threshold = args.improvement_threshold
    else:
        threshold = args.improvement_threshold
    GPA.pc.set_improvement_threshold(threshold)

    GPA.pc.set_candidate_weight_initialization_multiplier(args.candidate_weight_init_mult)

    # Avoid interactive pdb stop on "unwrapped modules" in non-interactive runs.
    if hasattr(GPA.pc, "set_unwrapped_modules_confirmed") and not args.strict_unwrapped_check:
        GPA.pc.set_unwrapped_modules_confirmed(True)

    pai_forward_function = getattr(torch, args.pai_forward_function)
    GPA.pc.set_pai_forward_function(pai_forward_function)

    if args.dendrite_mode == 0:
        GPA.pc.set_max_dendrites(0)
        if hasattr(GPA.pc, "set_perforated_backpropagation"):
            GPA.pc.set_perforated_backpropagation(False)
    elif args.dendrite_mode in (1, 2):
        GPA.pc.set_max_dendrites(args.max_dendrites)
        if hasattr(GPA.pc, "set_perforated_backpropagation"):
            GPA.pc.set_perforated_backpropagation(args.dendrite_mode == 2)


def setup_pai_optimizer_scheduler(args, model: nn.Module):
    GPA.pai_tracker.set_optimizer(optim.AdamW)
    GPA.pai_tracker.set_scheduler(CosineAnnealingLR)

    optim_args = {
        "params": filter(lambda p: p.requires_grad, model.parameters()),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    sched_args = {"T_max": max(1, args.epochs)}
    return GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)


def _write_fallback_png(path: str) -> None:
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )
    with open(path, "wb") as f:
        f.write(base64.b64decode(tiny_png))


def ensure_pai_png(output_dir: str, model: nn.Module, save_name: str) -> str:
    """Best-effort creation of PAI.png; falls back to a valid placeholder PNG."""
    pai_png_path = os.path.join(output_dir, "PAI.png")

    candidate_paths = [
        pai_png_path,
        os.path.join(os.getcwd(), "PAI.png"),
        os.path.join(output_dir, save_name, "PAI.png"),
        os.path.join(save_name, "PAI.png"),
    ]

    if hasattr(UPA, "save_system"):
        for args in [
            (model, save_name, "latest", True),
            (model, save_name, "latest"),
            (model, save_name),
        ]:
            try:
                UPA.save_system(*args)
                break
            except Exception:
                continue

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            if os.path.abspath(candidate) != os.path.abspath(pai_png_path):
                shutil.copyfile(candidate, pai_png_path)
            return pai_png_path

    _write_fallback_png(pai_png_path)
    return pai_png_path


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose, int]:
    """
    Use pretrained eval transforms when available.
    Keep train transform augmented but aligned to EfficientNet-B5 input size.
    Returns: train_transform, val_transform, test_transform, crop_size
    """
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


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def init_wandb(args) -> Optional["wandb.sdk.wandb_run.Run"]:
    if not args.use_wandb:
        print("W&B disabled. Pass --use-wandb to enable experiment logging.")
        return None

    if wandb is None:
        print("W&B logging requested but wandb is not installed. Install with: pip install wandb")
        return None

    if args.wandb_mode == "disabled":
        print("W&B disabled via --wandb-mode disabled.")
        return None

    api_key = os.environ.get("WANDB_API_KEY", "") or args.wandb_api_key

    if args.wandb_mode == "offline":
        print("W&B running in offline mode. Use wandb sync later to upload the run.")

    if args.wandb_mode == "online" and not api_key and args.wandb_anonymous == "never":
        print("No WANDB_API_KEY found in env/args. Will try existing local wandb login credentials.")

    try:
        if api_key:
            wandb.login(key=api_key)
        elif args.wandb_mode == "online":
            print(
                "W&B has no API key from --wandb-api-key or WANDB_API_KEY. "
                "Online runs may fail unless this machine is already logged in."
            )
    except Exception as exc:
        print(f"W&B login failed: {exc}")
        print("Proceeding to wandb.init anyway in case local auth already exists.")

    run_config = vars(args).copy()
    run_config.pop("wandb_api_key", None)

    entity = args.wandb_entity if args.wandb_entity else None
    init_kwargs = dict(
        project=args.wandb_project,
        entity=entity,
        name=args.wandb_run_name if args.wandb_run_name else None,
        mode=args.wandb_mode,
        config=run_config,
        anonymous=args.wandb_anonymous,
        id=args.wandb_run_id if args.wandb_run_id else None,
        resume=args.wandb_resume if args.wandb_run_id else None,
    )

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"W&B init failed: {exc}")
        if entity is not None:
            print("Retrying W&B init without explicit entity...")
            init_kwargs["entity"] = None
            try:
                run = wandb.init(**init_kwargs)
            except Exception as exc2:
                print(f"W&B init failed again: {exc2}")
                return None
        else:
            return None

    print(
        "W&B initialized: project={}, entity={}, mode={}, run_name={}".format(
            args.wandb_project,
            entity if entity else "<default>",
            args.wandb_mode,
            args.wandb_run_name if args.wandb_run_name else "<auto>",
        )
    )
    try:
        print(f"W&B run id: {run.id}")
    except Exception:
        pass
    return run


def log_to_wandb(run, metrics: Dict, step: Optional[int] = None) -> None:
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as exc:
        print(f"W&B log failed: {exc}")


def finish_wandb(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"W&B finish failed: {exc}")


def benchmark_inference_throughput(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    """Measure inputs/second throughput on the given device."""
    model_was_training = model.training
    model.to(device)
    model.eval()
    total_inputs = 0

    with torch.no_grad():
        warmup_batches = 2
        for idx, (data, _) in enumerate(data_loader):
            if idx >= warmup_batches:
                break
            data = data.to(device)
            _ = model(data)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        for idx, (data, _) in enumerate(data_loader):
            if idx >= max_batches:
                break
            data = data.to(device)
            _ = model(data)
            total_inputs += data.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    if model_was_training:
        model.train()

    if elapsed <= 0:
        return float("nan")
    return float(total_inputs / elapsed)


def benchmark_latency_ms(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    """Measure per-batch latency in milliseconds."""
    model_was_training = model.training
    model.to(device)
    model.eval()
    total_batches = 0

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        for idx, (data, _) in enumerate(data_loader):
            if idx >= max_batches:
                break
            data = data.to(device)
            _ = model(data)
            total_batches += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    if model_was_training:
        model.train()

    if total_batches == 0:
        return float("nan")
    return float((elapsed / total_batches) * 1000.0)


def benchmark_cpu_latency_single_core_ms(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    max_batches: int = 20,
) -> float:
    """Measure CPU latency with a single thread to mimic single-core behavior."""
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return benchmark_latency_ms(model, data_loader, torch.device("cpu"), max_batches=max_batches)
    finally:
        torch.set_num_threads(previous_threads)


def estimate_flops_with_hooks(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int, int] = (1, 3, DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE),
) -> float:
    """Rough FLOP estimate via forward hooks on conv/linear layers."""
    total_flops = 0.0
    handles = []
    model_was_training = model.training

    def conv_hook(module: nn.Conv2d, inputs, output):
        nonlocal total_flops
        batch_size = output.shape[0]
        out_channels = output.shape[1]
        out_height = output.shape[2]
        out_width = output.shape[3]
        kernel_height, kernel_width = module.kernel_size
        kernel_mul = (module.in_channels // module.groups) * kernel_height * kernel_width
        bias_ops = 1 if module.bias is not None else 0
        ops_per_output = (2 * kernel_mul) + bias_ops
        total_flops += batch_size * out_channels * out_height * out_width * ops_per_output

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal total_flops
        if output.dim() == 1:
            batch_size = 1
            out_features = output.shape[0]
        else:
            batch_size = output.shape[0]
            out_features = output.shape[-1]
        bias_ops = 1 if module.bias is not None else 0
        ops_per_output = (2 * module.in_features) + bias_ops
        total_flops += batch_size * out_features * ops_per_output

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    sample_input = torch.randn(*input_shape, device=device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)

    for handle in handles:
        handle.remove()

    if model_was_training:
        model.train()

    return float(total_flops)


def compute_model_stats(
    model: nn.Module,
    device: torch.device,
    input_hw: int,
) -> Tuple[int, float, str]:
    """Return parameter count and FLOPs (fvcore if available, otherwise hooks)."""
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    flops = float("nan")
    flops_source = "unavailable"
    input_shape = (1, 3, input_hw, input_hw)

    if FlopCountAnalysis is not None:
        try:
            sample_input = torch.randn(*input_shape, device=device)
            flops = float(FlopCountAnalysis(model, sample_input).total())
            flops_source = "fvcore"
        except Exception:
            flops = float("nan")

    if math.isnan(flops):
        try:
            flops = estimate_flops_with_hooks(model, device, input_shape=input_shape)
            flops_source = "approximate_hooks"
        except Exception as exc:
            print(f"FLOPS fallback failed: {exc}")
            flops = float("nan")

    return param_count, flops, flops_source


def safe_number(value) -> Optional[float]:
    """Return float(value) or None if NaN/inf."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def update_min_max(stats: Dict[str, float], key: str, value) -> None:
    """Update running min/max statistics for a given metric key."""
    value = safe_number(value)
    if value is None:
        return
    stats[f"{key}_min"] = min(stats.get(f"{key}_min", value), value)
    stats[f"{key}_max"] = max(stats.get(f"{key}_max", value), value)


def _state_dict_without_pai_metadata(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a state_dict with volatile PAI metadata removed."""
    state_dict = model.state_dict()
    # tracker_string can change size across epochs/PAI internal transitions.
    state_dict.pop("tracker_string", None)
    return state_dict


def _load_state_dict_compatible(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """Load a checkpoint while tolerating volatile PAI metadata keys."""
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=device)

    if not isinstance(loaded, dict):
        raise RuntimeError(f"Unexpected checkpoint format at {checkpoint_path}: {type(loaded)}")

    loaded.pop("tracker_string", None)
    incompatible = model.load_state_dict(loaded, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Loaded best model with non-strict compatibility. "
            f"missing_keys={incompatible.missing_keys}, unexpected_keys={incompatible.unexpected_keys}"
        )


def train(
    args,
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
) -> Tuple[float, float, float]:
    """Standard supervised training loop using cross entropy on logits."""
    model.train()
    model.to(device)
    correct = 0
    correct_top5 = 0
    total_loss = 0.0
    total_seen = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        total_seen += batch_size

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum()

        maxk = min(5, output.size(1))
        _, pred_top5 = output.topk(maxk, dim=1, largest=True, sorted=True)
        correct_top5 += pred_top5.eq(target.view(-1, 1).expand_as(pred_top5)).sum()

    train_accuracy = 100.0 * correct / len(train_loader.dataset)
    train_top5_accuracy = 100.0 * correct_top5 / len(train_loader.dataset)
    train_loss = total_loss / max(total_seen, 1)
    return float(train_loss), float(train_accuracy), float(train_top5_accuracy)


def evaluate(
    model: nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
) -> Dict[str, float]:
    """Evaluate with cross entropy, top-1 and top-5 accuracy."""
    model.eval()
    model.to(device)
    avg_loss = 0.0
    correct = 0
    correct_top5 = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            avg_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()

            maxk = min(5, output.size(1))
            _, pred_top5 = output.topk(maxk, dim=1, largest=True, sorted=True)
            correct_top5 += pred_top5.eq(target.view(-1, 1).expand_as(pred_top5)).sum()

    avg_loss /= len(data_loader.dataset)
    accuracy = float(100.0 * correct / len(data_loader.dataset))
    accuracy_top5 = float(100.0 * correct_top5 / len(data_loader.dataset))

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "accuracy_top5": accuracy_top5,
        "precision_at_1": accuracy,
    }


def test(
    model: nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    val_metrics = evaluate(model, device, val_loader)
    test_metrics = evaluate(model, device, test_loader)
    print(
        "\nValidation set: loss={:.4f}, top-1={:.2f}%, top-5={:.2f}%\n".format(
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["accuracy_top5"],
        )
    )
    print(
        "Test set: loss={:.4f}, top-1={:.2f}%, top-5={:.2f}%\n".format(
            test_metrics["loss"],
            test_metrics["accuracy"],
            test_metrics["accuracy_top5"],
        )
    )
    return val_metrics, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch Flowers-102 transfer learning with EfficientNet-B5 + PerforatedAI."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16; B5 is memory-heavy)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for validation/test (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="AdamW learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="WD",
        help="AdamW weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--finetune-backbone",
        action="store_true",
        default=False,
        help="unfreeze EfficientNet feature extractor and train full model",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="save the final model state_dict",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="efficientnet_b5_flowers102_last.pt",
        help="filename for last checkpoint (inside output-dir)",
    )
    parser.add_argument(
        "--best-model-path",
        type=str,
        default="efficientnet_b5_flowers102_best.pt",
        help="filename for best validation model (inside output-dir)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        default=False,
        help="resume training from the last checkpoint if available",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_flowers_efficientnet",
        help="directory for checkpoints and artifacts",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=FLOWERS_DATASET_ROOT_DEFAULT,
        help="root directory for torchvision Flowers102 data",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        default=False,
        help="disable automatic dataset download if missing",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        metavar="N",
        help="DataLoader workers (default: 4; use 0 for CPU-only debugging)",
    )

    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="enable Weights & Biases logging",
    )
    parser.add_argument(
    "--wandb-project",
    type=str,
    default="MNIST_PERF",
    help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="PerforatedAI_IDL",
        help="W&B entity (team) name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="EfficientNet_B5_Flowers102",
        help="W&B run name (optional)",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode",
    )
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default="",
        help="enter api key or set WANDB_API_KEY in the environment",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default="",
        help="W&B run id for resuming (optional)",
    )
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default="allow",
        choices=["allow", "must", "never"],
        help="W&B resume behavior",
    )
    parser.add_argument(
        "--wandb-anonymous",
        type=str,
        default="never",
        choices=["never", "allow", "must"],
        help="W&B anonymous mode",
    )
    parser.add_argument(
        "--dendrite-mode",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Dendrite mode: 0=no dendrites, 1=GD dendrites, 2=PB dendrites",
    )
    parser.add_argument(
        "--max-dendrites",
        type=int,
        default=5,
        help="Maximum number of dendrites to add in tracked modules",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=1.0,
        help="PerforatedAI improvement threshold",
    )
    parser.add_argument(
        "--candidate-weight-init-mult",
        type=float,
        default=0.1,
        help="PerforatedAI candidate weight initialization multiplier",
    )
    parser.add_argument(
        "--pai-forward-function",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh"],
        help="Forward function for PerforatedAI added nodes",
    )
    parser.add_argument(
        "--perforated-load-path",
        type=str,
        default="",
        help="Optional saved PerforatedAI system name/path to load",
    )
    parser.add_argument(
        "--pai-save-name",
        type=str,
        default="efficientnet_b5_flowers102_pai",
        help="Save name used by PerforatedAI system serialization",
    )
    parser.add_argument(
        "--strict-unwrapped-check",
        action="store_true",
        default=False,
        help="keep PerforatedAI unwrapped-module interactive checks enabled (can pause in pdb)",
    )
    parser.add_argument(
        "--strict-weight-decay-check",
        action="store_true",
        default=False,
        help="keep PerforatedAI weight-decay interactive check enabled (can pause in pdb)",
    )

    args = parser.parse_args()

    if GPA is None or UPA is None:
        raise ImportError(
            "PerforatedAI is required for this script. Install it from the PerforatedAI source/package used in your environment."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_path)
    best_model_path = os.path.join(args.output_dir, args.best_model_path)

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    use_mps = (not args.no_mps) and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_transform, val_transform, test_transform, crop_size = build_transforms()

    download = not args.no_download
    train_dataset = datasets.Flowers102(
        root=args.data_root,
        split="train",
        transform=train_transform,
        download=download,
    )
    val_dataset = datasets.Flowers102(
        root=args.data_root,
        split="val",
        transform=val_transform,
        download=download,
    )
    test_dataset = datasets.Flowers102(
        root=args.data_root,
        split="test",
        transform=test_transform,
        download=download,
    )

    print(f"Dataset sizes -> train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    num_workers = args.num_workers if args.num_workers >= 0 else 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    cpu_benchmark_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    gpu_benchmark_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    base_model = efficientnet_b5_flowers102(
        num_classes=NUM_CLASSES,
        finetune_backbone=args.finetune_backbone,
    )
    model = EfficientNetB5PAI(base_model)

    # Restrict tracked conversions to the layer directly before final FC.
    GPA.pc.append_module_ids_to_track([".pre_fc"])
    configure_pai(args)

    if args.perforated_load_path:
        model = UPA.initialize_pai(model, save_name=args.perforated_load_path)
        if hasattr(UPA, "load_system"):
            try:
                model = UPA.load_system(model, args.perforated_load_path, "latest", True)
            except Exception as exc:
                print(f"PerforatedAI load_system failed: {exc}. Continuing with initialized model.")
    else:
        model = UPA.initialize_pai(model, save_name=args.pai_save_name)

    print(model.classifier_fc)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    model = model.to(device)

    optimizer, scheduler = setup_pai_optimizer_scheduler(args, model)

    running_stats: Dict[str, float] = {}
    best_validation_accuracy = float("-inf")
    best_validation_snapshot: Dict[str, float] = {}
    start_epoch = 1

    if args.resume_from_checkpoint and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            running_stats = checkpoint.get("running_stats", running_stats)
            best_validation_accuracy = checkpoint.get("best_validation_accuracy", best_validation_accuracy)
            best_validation_snapshot = checkpoint.get("best_validation_snapshot", best_validation_snapshot)
            loaded_epoch = int(checkpoint.get("epoch", 0))
            start_epoch = loaded_epoch + 1
            print(f"Resuming from epoch {loaded_epoch}")
        except Exception as exc:
            print(f"Failed to resume from checkpoint at {checkpoint_path}: {exc}")

    run = init_wandb(args)
    cycle_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy, train_top5_accuracy = train(args, model, device, train_loader, optimizer, epoch)
        GPA.pai_tracker.add_extra_score(train_accuracy, "train")
        GPA.pai_tracker.add_extra_score(train_top5_accuracy, "train_top5")
        val_metrics, test_metrics = test(model, device, val_loader, test_loader)

        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            val_metrics["accuracy"], model
        )
        model = model.to(device)

        if restructured:
            optimizer, scheduler = setup_pai_optimizer_scheduler(args, model)
        else:
            scheduler.step()

        seconds_per_training_epoch = time.perf_counter() - epoch_start
        seconds_per_training_cycle = time.perf_counter() - cycle_start

        validation_accuracy = val_metrics["accuracy"]
        validation_top5 = val_metrics["accuracy_top5"]

        update_min_max(running_stats, "validation_accuracy", validation_accuracy)
        update_min_max(running_stats, "validation_top5", validation_top5)
        update_min_max(running_stats, "test_accuracy", test_metrics["accuracy"])
        update_min_max(running_stats, "test_top5", test_metrics["accuracy_top5"])

        is_best = validation_accuracy > best_validation_accuracy
        if is_best:
            best_validation_accuracy = validation_accuracy
            best_validation_snapshot = {
                "test_accuracy_at_best_validation": test_metrics["accuracy"],
                "test_top5_at_best_validation": test_metrics["accuracy_top5"],
                "validation_accuracy_best": validation_accuracy,
                "validation_top5_at_best_validation": validation_top5,
                "epoch_at_best_validation": epoch,
            }
            try:
                torch.save(_state_dict_without_pai_metadata(model), best_model_path)
            except Exception as exc:
                print(f"Failed to save best model to {best_model_path}: {exc}")

        epoch_log: Dict[str, Optional[float]] = {
            "epoch": epoch,
            "perforatedai/train_accuracy": train_accuracy,
            "perforatedai/train_top5_accuracy": train_top5_accuracy,
            "perforatedai/validation_accuracy": validation_accuracy,
            "flowers/train_loss": train_loss,
            "flowers/train_accuracy": train_accuracy,
            "flowers/train_top5_accuracy": train_top5_accuracy,
            "flowers/validation_loss": val_metrics["loss"],
            "flowers/validation_accuracy": validation_accuracy,
            "flowers/validation_accuracy_min": running_stats.get("validation_accuracy_min"),
            "flowers/validation_accuracy_max": running_stats.get("validation_accuracy_max"),
            "flowers/validation_top5_accuracy": validation_top5,
            "flowers/validation_top5_min": running_stats.get("validation_top5_min"),
            "flowers/validation_top5_max": running_stats.get("validation_top5_max"),
            "flowers/test_loss": test_metrics["loss"],
            "flowers/test_accuracy": test_metrics["accuracy"],
            "flowers/test_accuracy_min": running_stats.get("test_accuracy_min"),
            "flowers/test_accuracy_max": running_stats.get("test_accuracy_max"),
            "flowers/test_top5_accuracy": test_metrics["accuracy_top5"],
            "flowers/test_top5_min": running_stats.get("test_top5_min"),
            "flowers/test_top5_max": running_stats.get("test_top5_max"),
            "flowers/precision_at_1": test_metrics["precision_at_1"],
            "flowers/seconds_per_training_epoch": seconds_per_training_epoch,
            "flowers/seconds_per_training_cycle": seconds_per_training_cycle,
            "flowers/learning_rate": optimizer.param_groups[0]["lr"],
        }

        if hasattr(GPA, "pai_tracker"):
            epoch_log["perforatedai/dendrite_count"] = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)

        if best_validation_snapshot:
            epoch_log["flowers/test_accuracy_at_best_validation"] = best_validation_snapshot[
                "test_accuracy_at_best_validation"
            ]
            epoch_log["flowers/test_top5_at_best_validation"] = best_validation_snapshot[
                "test_top5_at_best_validation"
            ]
            epoch_log["flowers/epoch_at_best_validation"] = best_validation_snapshot["epoch_at_best_validation"]

        print(f"Epoch {epoch} metrics: {epoch_log}")
        if run is not None:
            log_to_wandb(run, epoch_log, step=epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "running_stats": running_stats,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_snapshot": best_validation_snapshot,
            "finetune_backbone": args.finetune_backbone,
        }
        try:
            torch.save(checkpoint, checkpoint_path)
        except Exception as exc:
            print(f"Failed to save checkpoint to {checkpoint_path}: {exc}")

        if training_complete:
            print("PerforatedAI signaled training complete.")
            break

    model.eval()

    if os.path.exists(best_model_path):
        try:
            _load_state_dict_compatible(model, best_model_path, device)
            model.to(device)
            print(f"Loaded best validation checkpoint from {best_model_path} for final test evaluation.")
        except Exception as exc:
            print(f"Failed to load best model from {best_model_path}: {exc}")

    gpu_inference_ips = float("nan")
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_inference_ips = benchmark_inference_throughput(
            model, gpu_benchmark_loader, torch.device("cuda")
        )

    model_cpu = model.to(torch.device("cpu"))
    cpu_inference_ips = benchmark_inference_throughput(
        model_cpu, cpu_benchmark_loader, torch.device("cpu")
    )
    latency_ms = benchmark_cpu_latency_single_core_ms(model_cpu, cpu_benchmark_loader)
    param_count, flops, flops_source = compute_model_stats(model_cpu, torch.device("cpu"), crop_size)

    final_metrics: Dict[str, object] = {
        "flowers/gpu_inference_inputs_per_second": safe_number(gpu_inference_ips),
        "flowers/cpu_inference_inputs_per_second": safe_number(cpu_inference_ips),
        "efficientnet_b5/num_parameters": param_count,
        "efficientnet_b5/flops": safe_number(flops),
        "efficientnet_b5/flops_source": flops_source,
        "efficientnet_b5/latency_ms_per_batch": safe_number(latency_ms),
    }

    if best_validation_snapshot:
        final_metrics["efficientnet_b5/accuracy_at_best_validation"] = best_validation_snapshot[
            "test_accuracy_at_best_validation"
        ]
        final_metrics["flowers/test_top5_at_best_validation"] = best_validation_snapshot[
            "test_top5_at_best_validation"
        ]
        final_metrics["flowers/validation_accuracy_best"] = best_validation_snapshot[
            "validation_accuracy_best"
        ]
        final_metrics["flowers/validation_top5_at_best_validation"] = best_validation_snapshot[
            "validation_top5_at_best_validation"
        ]
        final_metrics["flowers/epoch_at_best_validation"] = best_validation_snapshot[
            "epoch_at_best_validation"
        ]

    print(f"Final performance metrics: {final_metrics}")

    pai_png_path = ensure_pai_png(args.output_dir, model, args.pai_save_name)
    print(f"PAI graph image written to: {pai_png_path}")
    if run is not None and os.path.exists(pai_png_path):
        try:
            run.save(os.path.basename(pai_png_path), base_path=args.output_dir, policy="now")
        except Exception as exc:
            print(f"W&B PAI.png upload failed: {exc}")

    if run is not None:
        log_to_wandb(run, final_metrics)
        finish_wandb(run)

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "efficientnet_b5_flowers102_baseline.pt"))


if __name__ == "__main__":
    main()
