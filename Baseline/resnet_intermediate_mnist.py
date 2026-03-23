from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

try:
    import wandb
except ImportError:
    wandb = None

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


_AUC_WARNING_EMITTED = False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class MNISTResNet(nn.Module):
    def __init__(
        self,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        channels: Tuple[int, int, int, int] = (80, 160, 320, 640),
        num_classes: int = 10,
    ):
        super().__init__()
        self.inplanes = channels[0]

        # MNIST stem
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        # ResNet-18 layout
        self.layer1 = self._make_layer(channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_mnist() -> nn.Module:
    return MNISTResNet(
        layers=(2, 2, 2, 2),
        channels=(80, 160, 320, 640),
        num_classes=10,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        print(
            "No WANDB_API_KEY found in env/args. Will try existing local wandb login credentials."
        )

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


def compute_multiclass_auc(targets: torch.Tensor, probabilities: torch.Tensor) -> float:
    """
    Compute macro AUC over classes using one-vs-rest strategy.
    Expects probabilities (after softmax), not logits.
    """
    global _AUC_WARNING_EMITTED

    if roc_auc_score is None:
        if not _AUC_WARNING_EMITTED:
            print("AUC unavailable: scikit-learn is not installed in this environment.")
            _AUC_WARNING_EMITTED = True
        return float("nan")

    targets_np = targets.cpu().numpy()
    probs_np = probabilities.cpu().numpy()
    num_classes = probabilities.size(1)
    class_aucs = []
    present_classes = sorted(set(int(value) for value in targets_np.tolist()))

    try:
        for class_index in range(num_classes):
            binary_targets = (targets_np == class_index).astype(int)
            positive_count = int(binary_targets.sum())
            negative_count = int(binary_targets.shape[0] - positive_count)

            if positive_count == 0 or negative_count == 0:
                continue

            class_auc = roc_auc_score(binary_targets, probs_np[:, class_index])
            if not math.isnan(float(class_auc)):
                class_aucs.append(float(class_auc))

        if class_aucs:
            return float(sum(class_aucs) / len(class_aucs))

        if not _AUC_WARNING_EMITTED:
            print(
                "AUC unavailable: no valid one-vs-rest class AUCs could be computed. "
                "present_classes={}, probability_shape={}, probability_min={}, probability_max={}".format(
                    present_classes,
                    probs_np.shape,
                    float(probs_np.min()),
                    float(probs_np.max()),
                )
            )
            _AUC_WARNING_EMITTED = True
        return float("nan")
    except ValueError as exc:
        if not _AUC_WARNING_EMITTED:
            print(
                "AUC unavailable: {}. present_classes={}, probability_shape={}, probability_min={}, probability_max={}".format(
                    exc,
                    present_classes,
                    probs_np.shape,
                    float(probs_np.min()),
                    float(probs_np.max()),
                )
            )
            _AUC_WARNING_EMITTED = True
        return float("nan")
    except Exception as exc:
        if not _AUC_WARNING_EMITTED:
            print(
                "AUC unavailable due to unexpected error: {}. present_classes={}, probability_shape={}, probability_min={}, probability_max={}".format(
                    exc,
                    present_classes,
                    probs_np.shape,
                    float(probs_np.min()),
                    float(probs_np.max()),
                )
            )
            _AUC_WARNING_EMITTED = True
        return float("nan")


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
    input_shape: Tuple[int, int, int, int] = (1, 1, 28, 28),
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


def compute_model_stats(model: nn.Module, device: torch.device) -> Tuple[int, float, str]:
    """Return parameter count and FLOPs (fvcore if available, otherwise hooks)."""
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    flops = float("nan")
    flops_source = "unavailable"

    if FlopCountAnalysis is not None:
        try:
            sample_input = torch.randn(1, 1, 28, 28, device=device)
            flops = float(FlopCountAnalysis(model, sample_input).total())
            flops_source = "fvcore"
        except Exception:
            flops = float("nan")

    if math.isnan(flops):
        try:
            flops = estimate_flops_with_hooks(model, device)
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


def train(
    args,
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
) -> float:
    """Standard supervised training loop using cross entropy on logits."""
    model.train()
    model.to(device)
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)  # logits
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

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

    train_accuracy = 100.0 * correct.item() / len(train_loader.dataset)
    return train_accuracy


def evaluate(
    model: nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
) -> Dict[str, float]:
    """Evaluate model with cross entropy and compute accuracy, AUC, precision@1."""
    model.eval()
    model.to(device)
    avg_loss = 0.0
    correct = 0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # logits
            probs = torch.softmax(output, dim=1)

            avg_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()

            all_targets.append(target.detach().cpu())
            all_probs.append(probs.detach().cpu())

    avg_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct.item() / len(data_loader.dataset)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    auc = compute_multiclass_auc(all_targets, all_probs)
    precision_at_1 = accuracy  # for single-label classification, Precision@1 == top-1 accuracy

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
        "precision_at_1": precision_at_1,
    }
    return metrics


def test(
    model: nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Simple evaluation helper:
    - no model restructuring
    - no tracker logic
    - no optimizer reset
    """
    val_metrics = evaluate(model, device, val_loader)
    test_metrics = evaluate(model, device, test_loader)

    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {:.2f}%, AUC: {}\n".format(
            val_metrics["loss"],
            val_metrics["accuracy"],
            "nan" if safe_number(val_metrics["auc"]) is None else "{:.6f}".format(val_metrics["auc"]),
        )
    )
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, AUC: {}\n".format(
            test_metrics["loss"],
            test_metrics["accuracy"],
            "nan" if safe_number(test_metrics["auc"]) is None else "{:.6f}".format(test_metrics["auc"]),
        )
    )

    return val_metrics, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch MNIST ResNet-18 intermediate (no perforation).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        metavar="N",
        help="fraction of train set used for validation (default: 0.1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="learning rate step gamma (default: 0.7)",
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
        help="For saving the current model",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="resnet18_intermediate_last.pt",
        help="Filename for last checkpoint (inside output-dir).",
    )
    parser.add_argument(
        "--best-model-path",
        type=str,
        default="resnet18_intermediate_best.pt",
        help="Filename for best-accuracy model (inside output-dir).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        default=False,
        help="Resume training from the last checkpoint if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for checkpoints and artifacts.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/ocean/projects/cis260045p/shared/data",
        help="Root directory for MNIST data (PSC default or local path).",
    )
    
    # W&B arguments mirror Olivia's script for easy comparison
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
        default="",
        help="W&B entity (team) name (optional)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="",
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

    args = parser.parse_args()

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

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    data_root = args.data_root
    try:
        train_full = datasets.MNIST(
            data_root,
            train=True,
            download=False,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            data_root,
            train=False,
            download=False,
            transform=transform,
        )
    except RuntimeError:
        train_full = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            data_root,
            train=False,
            download=True,
            transform=transform,
        )

    total_train = len(train_full)
    val_size = int(total_train * args.val_split)
    val_size = max(1, val_size)
    train_size = total_train - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Explicit loader settings so CPU training is shuffled and val/test never are
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=1 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=1 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    # Benchmark loaders: CPU batch=1, GPU batch=100 on the test set
    cpu_benchmark_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    gpu_benchmark_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = resnet18_mnist()
    model = model.to(device)

    # print out model params + architecture
    print("Model architecture:")
    print(model)
    print()
    print("Model parameters:")
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

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

        train_accuracy = train(args, model, device, train_loader, optimizer, epoch)
        val_metrics, test_metrics = test(model, device, val_loader, test_loader)

        scheduler.step()

        seconds_per_training_epoch = time.perf_counter() - epoch_start
        seconds_per_training_cycle = time.perf_counter() - cycle_start

        validation_accuracy = val_metrics["accuracy"]
        validation_auc = val_metrics["auc"]

        update_min_max(running_stats, "validation_accuracy", validation_accuracy)
        update_min_max(running_stats, "validation_auc", validation_auc)
        update_min_max(running_stats, "test_accuracy", test_metrics["accuracy"])
        update_min_max(running_stats, "test_auc", test_metrics["auc"])

        is_best = validation_accuracy > best_validation_accuracy
        if is_best:
            best_validation_accuracy = validation_accuracy
            best_validation_snapshot = {
                "test_accuracy_at_best_validation": test_metrics["accuracy"],
                "test_auc_at_best_validation": test_metrics["auc"],
                "validation_accuracy_best": validation_accuracy,
                "validation_auc_at_best_validation": validation_auc,
                "epoch_at_best_validation": epoch,
            }
            try:
                torch.save(model.state_dict(), best_model_path)
            except Exception as exc:
                print(f"Failed to save best model to {best_model_path}: {exc}")

        epoch_log: Dict[str, Optional[float]] = {
            "epoch": epoch,
            "perforatedai/train_accuracy": train_accuracy,
            "perforatedai/validation_accuracy": validation_accuracy,
            "perforatedai/validation_accuracy_min": running_stats.get("validation_accuracy_min"),
            "perforatedai/validation_accuracy_max": running_stats.get("validation_accuracy_max"),
            "perforatedai/validation_auc": safe_number(validation_auc),
            "perforatedai/validation_auc_min": running_stats.get("validation_auc_min"),
            "perforatedai/validation_auc_max": running_stats.get("validation_auc_max"),
            "perforatedai/test_accuracy": test_metrics["accuracy"],
            "perforatedai/test_accuracy_min": running_stats.get("test_accuracy_min"),
            "perforatedai/test_accuracy_max": running_stats.get("test_accuracy_max"),
            "perforatedai/test_auc": safe_number(test_metrics["auc"]),
            "perforatedai/test_auc_min": running_stats.get("test_auc_min"),
            "perforatedai/test_auc_max": running_stats.get("test_auc_max"),
            "perforatedai/precision_at_1": test_metrics["precision_at_1"],
            "perforatedai/seconds_per_training_epoch": seconds_per_training_epoch,
            "perforatedai/seconds_per_training_cycle": seconds_per_training_cycle,
        }

        if best_validation_snapshot:
            epoch_log["perforatedai/test_accuracy_at_best_validation"] = best_validation_snapshot[
                "test_accuracy_at_best_validation"
            ]
            epoch_log["perforatedai/test_auc_at_best_validation"] = safe_number(
                best_validation_snapshot["test_auc_at_best_validation"]
            )
            epoch_log["perforatedai/epoch_at_best_validation"] = best_validation_snapshot["epoch_at_best_validation"]

        print(f"Epoch {epoch} metrics: {epoch_log}")
        if run is not None:
            log_to_wandb(run, epoch_log, step=epoch)

        # Save last checkpoint each epoch.
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "running_stats": running_stats,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_snapshot": best_validation_snapshot,
        }
        try:
            torch.save(checkpoint, checkpoint_path)
        except Exception as exc:
            print(f"Failed to save checkpoint to {checkpoint_path}: {exc}")

    model.eval()

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
    param_count, flops, flops_source = compute_model_stats(model_cpu, torch.device("cpu"))

    final_metrics: Dict[str, object] = {
        "perforatedai/gpu_inference_inputs_per_second": safe_number(gpu_inference_ips),
        "perforatedai/cpu_inference_inputs_per_second": safe_number(cpu_inference_ips),
        "resnet/num_parameters": param_count,
        "resnet/flops": safe_number(flops),
        "resnet/flops_source": flops_source,
        "resnet/latency_ms_per_batch": safe_number(latency_ms),
    }

    if best_validation_snapshot and safe_number(latency_ms) is not None:
        final_metrics["resnet/accuracy_at_best_validation"] = best_validation_snapshot[
            "test_accuracy_at_best_validation"
        ]
        final_metrics["perforatedai/test_auc_at_best_validation"] = safe_number(
            best_validation_snapshot["test_auc_at_best_validation"]
        )
        final_metrics["perforatedai/validation_accuracy_best"] = best_validation_snapshot[
            "validation_accuracy_best"
        ]
        final_metrics["perforatedai/validation_auc_at_best_validation"] = safe_number(
            best_validation_snapshot["validation_auc_at_best_validation"]
        )
        final_metrics["perforatedai/epoch_at_best_validation"] = best_validation_snapshot[
            "epoch_at_best_validation"
        ]

    print(f"Final performance metrics: {final_metrics}")
    if run is not None:
        log_to_wandb(run, final_metrics)
        finish_wandb(run)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_resnet18_intermediate.pt")


if __name__ == "__main__":
    main()

# Intermediate-width ResNet-18 MNIST for comparison against perforated runs
