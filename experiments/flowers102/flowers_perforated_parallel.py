'''
For running:
interact -p GPU-shared --gres=gpu:v100-32:2 -t 8:00:00 -A cis260045p
python -m torch.distributed.run --nproc_per_node 2 /ocean/projects/cis260045p/shared/flowers_perforated_parallel.py --data-root ./data --use-wandb --wandb-api-key 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 --dendrite-mode 1 --max-dendrites 5 --pai-forward-function relu --improvement-threshold 0.5 --candidate-weight-init-mult 0.1 --epochs 40 > output.txt 2>&1
'''

from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import math
import os
import pdb
import shutil
import sys
import time
import warnings
from typing import Dict, Tuple, Optional, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
DEFAULT_PAI_CONVERT_TARGET = "pre_fc"


def disable_interactive_breakpoints() -> None:
    """Disable breakpoint()/pdb.set_trace() in non-interactive training runs."""
    os.environ.setdefault("PYTHONBREAKPOINT", "0")

    def _noop_breakpoint(*args, **kwargs):
        return None

    def _noop_exit(*args, **kwargs):
        return None

    builtins.breakpoint = _noop_breakpoint
    pdb.set_trace = _noop_breakpoint
    # Some external libraries call exit()/quit() from interactive checks.
    # Treat these as no-ops in batch training instead of terminating the job.
    builtins.exit = _noop_exit
    builtins.quit = _noop_exit
    sys.exit = _noop_exit


@contextmanager
def suppress_output() -> Generator[None, None, None]:
    """Temporarily silence stdout/stderr for noisy library calls."""
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


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


def _selected_convert_module_id(args) -> str:
    return ".classifier_fc" if args.pai_convert_target == "classifier_fc" else ".pre_fc"


def configure_pai(args, model: Optional[nn.Module] = None) -> None:
    """Configure PerforatedAI global parameters.

    Strategy: Use EXPLICIT module ID targeting only.
    - append_module_ids_to_convert([".pre_fc"]) to tell PAI ONLY this module gets dendrites
    - Do NOT use append_module_names_to_convert (type-based) as it converts ALL matching types

    This avoids the issue where PAI converts both .pre_fc AND .classifier_fc when
    we say "convert Linear layers" — we only want .pre_fc.
    """
    GPA.pc.set_testing_dendrite_capacity(False)

    convert_target = _selected_convert_module_id(args)  # ".pre_fc" or ".classifier_fc"

    # EXPLICIT: Tell PAI to convert ONLY this specific module ID
    GPA.pc.append_module_ids_to_convert([convert_target])

    # Verbose is left ON so PAI diagnostic messages appear in the log (helps debug).
    GPA.pc.set_verbose(True)

    # Avoid interactive pdb stop on weight-decay warnings in non-interactive runs.
    if hasattr(GPA.pc, "set_weight_decay_accepted") and not args.strict_weight_decay_check:
        GPA.pc.set_weight_decay_accepted(True)

    # Avoid interactive pdb stop on "unwrapped modules" in non-interactive runs.
    if hasattr(GPA.pc, "set_unwrapped_modules_confirmed") and not args.strict_unwrapped_check:
        GPA.pc.set_unwrapped_modules_confirmed(True)

    print(f"PAI: EXPLICIT module_ids_to_convert = ['{convert_target}']")
    print(f"PAI: No type-based filtering (module_names_to_convert not used)")

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
    model_for_optim = model.module if isinstance(model, DDP) else model

    GPA.pai_tracker.set_optimizer(optim.AdamW)
    GPA.pai_tracker.set_scheduler(CosineAnnealingLR)

    # Ensure at least the task head remains trainable after wrapping/loading.
    for name in ["pre_fc", "classifier_fc"]:
        if hasattr(model_for_optim, name):
            for p in getattr(model_for_optim, name).parameters():
                p.requires_grad = True

    trainable_params = [p for p in model_for_optim.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found (all layers are frozen). "
            "Enable trainable layers or run with --finetune-backbone."
        )

    optim_args = {
        "params": trainable_params,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    sched_args = {"T_max": max(1, args.epochs)}
    return GPA.pai_tracker.setup_optimizer(model_for_optim, optim_args, sched_args)


def _resolve_pai_save_name(output_dir: str, save_name: str) -> str:
    """Resolve save_name to a path rooted in output_dir unless already absolute."""
    normalized_save_name = os.path.normpath(save_name)
    normalized_output_dir = os.path.normpath(output_dir)

    if os.path.isabs(normalized_save_name):
        return normalized_save_name

    # If caller already provided a path under output_dir, do not prefix output_dir again.
    if normalized_save_name == normalized_output_dir or normalized_save_name.startswith(
        normalized_output_dir + os.sep
    ):
        return normalized_save_name

    return os.path.join(output_dir, normalized_save_name)


def _pai_path_variants(path: str) -> list[str]:
    """Return both absolute and cwd-relative variants for a PAI path."""
    normalized = os.path.normpath(path)
    variants = {normalized}

    if os.path.isabs(normalized):
        variants.add(normalized.lstrip(os.sep))
    else:
        variants.add(os.path.abspath(normalized))

    return [variant for variant in variants if variant]


def ensure_pai_png(output_dir: str, model: nn.Module, save_name: str = "PAI") -> str:
    """Create/copy PAI.png from PerforatedAI default artifact locations."""
    pai_png_path = os.path.join(output_dir, "PAI.png")
    resolved_save_name = save_name
    if os.path.dirname(resolved_save_name):
        os.makedirs(os.path.dirname(resolved_save_name), exist_ok=True)

    candidate_paths = [
        os.path.join(resolved_save_name, "PAI.png"),
        os.path.join(save_name, "PAI.png"),
        os.path.join(output_dir, save_name, "PAI.png"),
        pai_png_path,
        os.path.join(os.getcwd(), "PAI.png"),
        os.path.join(os.getcwd(), "PAI", "PAI.png"),
    ]
    # Keep order while removing duplicates.
    candidate_paths = list(dict.fromkeys(candidate_paths))

    save_errors = []
    if hasattr(UPA, "save_system"):
        for call_args in [
            (model, resolved_save_name, "latest"),
            (model, resolved_save_name),
        ]:
            try:
                with suppress_output():
                    UPA.save_system(*call_args)
                break
            except Exception as exc:
                save_errors.append(f"save_system{call_args[1:]} -> {exc}")
                continue

    for candidate in candidate_paths:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            if os.path.abspath(candidate) != os.path.abspath(pai_png_path):
                shutil.copyfile(candidate, pai_png_path)
            return pai_png_path

    checked = "\n  - ".join(candidate_paths)
    error_text = "\n".join(save_errors) if save_errors else "No save_system call succeeded or produced PAI.png."
    raise RuntimeError(
        "PAI.png was not generated by PerforatedAI. Checked paths:\n"
        f"  - {checked}\n"
        f"save_system errors:\n{error_text}"
    )


def ensure_pai_switch_files_exist(
    pai_system_name: str,
    model: nn.Module,
    *,
    is_distributed: bool,
    is_main_process: bool,
) -> None:
    """Ensure PerforatedAI switch-mode checkpoint files exist in a DDP-safe way."""
    pai_root = os.path.abspath(os.path.normpath(pai_system_name))
    os.makedirs(pai_root, exist_ok=True)

    best_model_path = os.path.join(pai_root, "best_model.pt")
    latest_path = os.path.join(pai_root, "latest.pt")

    # Own file creation on rank 0 to avoid races/no-op saves on non-owner ranks.
    if is_main_process:
        if not os.path.exists(latest_path) and hasattr(UPA, "save_system"):
            with suppress_output():
                UPA.save_system(model, pai_root, "latest")
        if not os.path.exists(best_model_path) and hasattr(UPA, "save_system"):
            with suppress_output():
                UPA.save_system(model, pai_root, "best_model")

        # Fallback for environments that only produced latest.
        if not os.path.exists(best_model_path) and os.path.exists(latest_path):
            shutil.copyfile(latest_path, best_model_path)

    if is_distributed:
        dist.barrier()

    if not os.path.exists(best_model_path):
        raise RuntimeError(f"Missing required PAI switch file after seeding: {best_model_path}")


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def setup_for_distributed(is_master: bool) -> None:
    """Disable printing when not in the master process."""
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process_rank() -> bool:
    return get_rank() == 0


def save_on_master(*args, **kwargs) -> None:
    if is_main_process_rank():
        torch.save(*args, **kwargs)


def init_distributed_mode(args) -> None:
    dist_url = getattr(args, "dist_url", "env://")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if dist_url == "env://":
            # Allow single-node launches that inherit RANK/WORLD_SIZE without a full rendezvous setup.
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", "0"))
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
        args.gpu = args.rank % max(torch.cuda.device_count(), 1)
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.local_rank = 0
        return

    args.distributed = True
    args.local_rank = args.gpu

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
    args.dist_url = dist_url
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


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


def init_wandb(args):
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


def _state_dict_without_pai_metadata(
    model: nn.Module,
    disallowed_module_id: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Return a state_dict with volatile PAI metadata removed."""
    state_dict = model.state_dict()
    # tracker_string can change size across epochs/PAI internal transitions.
    state_dict.pop("tracker_string", None)
    # Remove stale dendrite keys from modules that should not be converted.
    # This prevents PAI from trying to reconstruct them during switch_mode.
    keys_to_remove = []
    if disallowed_module_id:
        keys_to_remove = [
            k for k in state_dict.keys() if "dendrite_module" in k and disallowed_module_id in k
        ]
    for key in keys_to_remove:
        state_dict.pop(key, None)
        print(f"Excluded stale dendrite key from save: {key}")
    return state_dict


def _load_state_dict_compatible(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    disallowed_module_id: Optional[str] = None,
) -> None:
    """Load a checkpoint while tolerating volatile PAI metadata keys."""
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=device)

    if not isinstance(loaded, dict):
        raise RuntimeError(f"Unexpected checkpoint format at {checkpoint_path}: {type(loaded)}")

    # Remove volatile PAI metadata and any stale dendrite keys from untracked modules.
    loaded.pop("tracker_string", None)
    
    # Filter out stale dendrite keys from explicitly disallowed conversion modules.
    keys_to_remove = []
    if disallowed_module_id:
        keys_to_remove = [k for k in loaded.keys() if "dendrite_module" in k and disallowed_module_id in k]
    for key in keys_to_remove:
        loaded.pop(key, None)
        print(f"Removed stale dendrite key from loaded dict: {key}")
    
    incompatible = model.load_state_dict(loaded, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Loaded best model with non-strict compatibility. "
            f"missing_keys={incompatible.missing_keys}, unexpected_keys={incompatible.unexpected_keys}"
        )


def _assert_disallowed_module_not_converted(model: nn.Module, disallowed_module_id: Optional[str]) -> None:
    """Fail fast if PAI converted a module that should be excluded from conversion."""
    if not disallowed_module_id:
        return
    model_for_check = model.module if isinstance(model, DDP) else model
    state_keys = model_for_check.state_dict().keys()
    bad_prefixes = (
        f"{disallowed_module_id.lstrip('.')}.main_module.",
        f"{disallowed_module_id.lstrip('.')}.dendrite_module.",
    )
    converted = [k for k in state_keys if k.startswith(bad_prefixes)]
    if converted:
        sample = converted[:5]
        raise RuntimeError(
            f"PAI converted {disallowed_module_id}, which is incompatible with this training setup. "
            "This can cause missing dendrite values during switch_mode. "
            f"Example converted keys: {sample}"
        )


def _load_state_dict_into_model(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    disallowed_module_id: Optional[str] = None,
) -> None:
    """Load a state dict while tolerating volatile PAI metadata keys."""
    cleaned_state_dict = dict(state_dict)
    cleaned_state_dict.pop("tracker_string", None)
    keys_to_remove = []
    if disallowed_module_id:
        keys_to_remove = [
            k for k in cleaned_state_dict.keys() if "dendrite_module" in k and disallowed_module_id in k
        ]
    for key in keys_to_remove:
        cleaned_state_dict.pop(key, None)
    incompatible = model.load_state_dict(cleaned_state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Loaded model state with non-strict compatibility. "
            f"missing_keys={incompatible.missing_keys}, unexpected_keys={incompatible.unexpected_keys}"
        )


def _model_has_pai_modules(model: nn.Module) -> bool:
    """Best-effort check that the model is still PAI-wrapped."""
    model_for_check = model.module if isinstance(model, DDP) else model
    try:
        keys = model_for_check.state_dict().keys()
    except Exception:
        return False
    # Require concrete wrapper markers; tracker_string alone is not sufficient.
    markers = (".main_module.", ".dendrite_module.")
    return any(any(marker in key for marker in markers) for key in keys)


def _is_nonfatal_pai_system_exit(exc: BaseException) -> bool:
    """Treat known PAI interactive exit codes as non-fatal in batch training."""
    if not isinstance(exc, SystemExit):
        return False
    code = exc.code
    return code in (-1, 0, None, "-1", "0")


def _call_pai_add_validation_score(args, validation_accuracy: float, model: nn.Module):
    """Call PAI validation score hook.  Always visible so PAI diagnostics reach the log."""
    return GPA.pai_tracker.add_validation_score(validation_accuracy, model)


def recover_pai_model_if_needed(model: nn.Module, pai_system_name: str) -> nn.Module:
    """Re-initialize/load PAI when wrapper markers are missing."""
    if _model_has_pai_modules(model):
        return model

    model = UPA.initialize_pai(model, save_name=pai_system_name)
    latest_path = os.path.join(os.path.abspath(os.path.normpath(pai_system_name)), "latest.pt")
    if hasattr(UPA, "load_system") and os.path.exists(latest_path):
        try:
            model = UPA.load_system(model, pai_system_name, "latest", True)
        except BaseException as exc:
            print(f"PAI recovery load_system failed: {exc}. Continuing with initialized model.")
    return model


def prepare_pai_switch_model(
    model: nn.Module,
    pai_system_name: str,
    finetune_backbone: bool,
    disallowed_module_id: Optional[str] = None,
) -> nn.Module:
    """Build a fresh PAI-wrapped model for switch_mode from the current training weights."""
    model_for_switch = model.module if isinstance(model, DDP) else model
    source_state_dict = model_for_switch.state_dict()

    fresh_base_model = efficientnet_b5_flowers102(
        num_classes=NUM_CLASSES,
        finetune_backbone=finetune_backbone,
    )
    fresh_model = EfficientNetB5PAI(fresh_base_model)
    _load_state_dict_into_model(fresh_model, source_state_dict, disallowed_module_id=disallowed_module_id)

    fresh_model = UPA.initialize_pai(fresh_model, save_name=pai_system_name)
    latest_path = os.path.join(os.path.abspath(os.path.normpath(pai_system_name)), "latest.pt")
    if hasattr(UPA, "load_system") and os.path.exists(latest_path):
        try:
            fresh_model = UPA.load_system(fresh_model, pai_system_name, "latest", True)
        except BaseException as exc:
            print(f"PAI switch model load of latest failed: {exc}. Continuing with freshly initialized model.")
    return fresh_model


def train(
    args,
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    is_main_process: bool = True,
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

        if is_main_process and batch_idx % args.log_interval == 0:
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

    # In DDP, reduce train totals so every rank reports global train metrics.
    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor(
            [
                total_loss,
                float(correct.item()),
                float(correct_top5.item()),
                float(total_seen),
            ],
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, correct_value, correct_top5_value, total_seen = [float(x) for x in stats.tolist()]
    else:
        correct_value = float(correct.item())
        correct_top5_value = float(correct_top5.item())

    denom = max(int(total_seen), 1)
    train_accuracy = 100.0 * correct_value / denom
    train_top5_accuracy = 100.0 * correct_top5_value / denom
    train_loss = total_loss / denom
    return float(train_loss), float(train_accuracy), float(train_top5_accuracy)


def evaluate(
    model: nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
) -> Dict[str, float]:
    """Evaluate with cross entropy, top-1 and top-5 accuracy."""
    model.eval()
    model.to(device)
    total_loss = 0.0
    correct = 0.0
    correct_top5 = 0.0
    total_seen = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            batch_size = target.size(0)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            total_seen += batch_size
            pred = output.argmax(dim=1, keepdim=True)
            correct += float(pred.eq(target.view_as(pred)).sum().item())

            maxk = min(5, output.size(1))
            _, pred_top5 = output.topk(maxk, dim=1, largest=True, sorted=True)
            correct_top5 += float(pred_top5.eq(target.view(-1, 1).expand_as(pred_top5)).sum().item())

    # In DDP, reduce metrics so every rank sees identical validation/test values.
    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([total_loss, correct, correct_top5, float(total_seen)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, correct, correct_top5, total_seen = [float(x) for x in stats.tolist()]

    denom = max(int(total_seen), 1)
    avg_loss = total_loss / denom
    accuracy = float(100.0 * correct / denom)
    accuracy_top5 = float(100.0 * correct_top5 / denom)

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
    is_main_process: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    val_metrics = evaluate(model, device, val_loader)
    test_metrics = evaluate(model, device, test_loader)
    if is_main_process:
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


@record
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
        default=0,
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
        "--pai-convert-target",
        type=str,
        default=DEFAULT_PAI_CONVERT_TARGET,
        choices=["pre_fc", "classifier_fc"],
        help=(
            "Module to convert with dendrites. "
            "Default pre_fc targets layer n-1 (recommended for this PAI workflow)."
        ),
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
        help="PAI save name/path (relative paths are created under --output-dir)",
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
    parser.add_argument(
        "--debug-pai-switch-logs",
        action="store_true",
        default=False,
        help="show raw PerforatedAI output around add_validation_score for debugging",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action="store_true",
        default=False,
        help="enable DDP unused-parameter detection (adds autograd overhead)",
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )

    args = parser.parse_args()

    # Default to non-interactive behavior for long-running jobs.
    # Keep interactive breakpoints only when strict checks are explicitly enabled.
    if not args.strict_unwrapped_check and not args.strict_weight_decay_check:
        disable_interactive_breakpoints()

    # PerforatedAI internal scheduler paths can trigger this warning even when
    # the main training loop calls optimizer.step() before scheduler.step().
    warnings.filterwarnings(
        "ignore",
        module=r"torch\.optim\.lr_scheduler",
        category=UserWarning,
    )

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    init_distributed_mode(args)
    is_distributed = getattr(args, "distributed", False)
    is_main_process = is_main_process_rank()
    args.global_rank = getattr(args, "rank", 0)
    args.world_size = get_world_size()
    args.local_rank = getattr(args, "gpu", 0)

    if GPA is None or UPA is None:
        raise ImportError(
            "PerforatedAI is required for this script. Install it from the PerforatedAI source/package used in your environment."
        )

    convert_module_id = _selected_convert_module_id(args)
    # Historical switch-mode mismatch is only known for classifier_fc conversion in this script.
    disallowed_module_id = ".classifier_fc" if convert_module_id == ".pre_fc" else None
    if is_main_process and args.pai_convert_target != "pre_fc":
        print(
            "Warning: non-n-1 PAI conversion target selected "
            f"({args.pai_convert_target}). This mode is less stable in this workflow."
        )

    args.output_dir = os.path.abspath(args.output_dir)
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_path)
    best_model_path = os.path.join(args.output_dir, args.best_model_path)

    use_mps = (not args.no_mps) and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    # For DDP, set device to local rank; for single-GPU or CPU, use standard logic
    if is_distributed and use_cuda:
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
    elif use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if is_main_process:
        if is_distributed:
            print(f"Using DistributedDataParallel: rank {args.global_rank}/{args.world_size}, local_rank {args.local_rank}, device {device}")
            # Account for auto-enabling when dendrite_mode > 0
            effective_find_unused = args.ddp_find_unused_parameters or (args.dendrite_mode > 0)
            print(f"DDP find_unused_parameters={effective_find_unused} " + ("(auto-enabled for dendrite mode)" if args.dendrite_mode > 0 and not args.ddp_find_unused_parameters else ""))
        else:
            print(f"Single-process mode on device {device}")
        if args.debug_pai_switch_logs:
            print("PAI switch debug logs enabled (--debug-pai-switch-logs).")

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

    # For distributed training, use DistributedSampler; for single-rank, use standard shuffling
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.global_rank,
        shuffle=True,
        seed=args.seed,
    ) if is_distributed else None

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=args.global_rank,
        shuffle=False,
        seed=args.seed,
    ) if is_distributed else None

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=args.world_size,
        rank=args.global_rank,
        shuffle=False,
        seed=args.seed,
    ) if is_distributed else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if not using DistributedSampler
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    cpu_benchmark_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler if is_distributed else None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    gpu_benchmark_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        sampler=val_sampler if is_distributed else None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    base_model = efficientnet_b5_flowers102(
        num_classes=NUM_CLASSES,
        finetune_backbone=args.finetune_backbone,
    )
    model = EfficientNetB5PAI(base_model)

    # Configure PAI module selection: mark frozen modules to skip, restrict to Linear dendrites.
    configure_pai(args, model=model)

#     pai_system_name = (
#     args.perforated_load_path
#     if args.perforated_load_path
#     else _resolve_pai_save_name(args.output_dir, args.pai_save_name)
# )

# #  Clean old PAI artifacts ONLY if starting fresh
#     if not args.perforated_load_path:
#         if os.path.exists(pai_system_name):
#             print(f"Removing old PAI directory at {pai_system_name}...")
#             shutil.rmtree(pai_system_name)

#     # Ensure directories exist
#     os.makedirs(pai_system_name, exist_ok=True)
#     os.makedirs(os.path.join(pai_system_name, args.output_dir), exist_ok=True)

#     #  Initialize / Load PerforatedAI system
#     if args.perforated_load_path:
#         print(f"Loading existing PAI system from: {pai_system_name}")
        
#         model = UPA.initialize_pai(model, save_name=pai_system_name)

#         if hasattr(UPA, "load_system"):
#             try:
#                 model = UPA.load_system(model, pai_system_name, "latest", True)
#                 print("Successfully loaded PAI system.")
#             except Exception as exc:
#                 print(f"PerforatedAI load_system failed: {exc}")
#                 print("Continuing with freshly initialized PAI model.")
#     else:
#         print(f"Initializing new PAI system at: {pai_system_name}")
#         model = UPA.initialize_pai(model, save_name=pai_system_name)

#     print("Final classifier layer:")
#     print(model.classifier_fc)
    # Use a stable, run-specific save path by default so fresh runs do not reuse stale PAI state.
    pai_system_name = os.path.abspath(
        args.perforated_load_path
        if args.perforated_load_path
        else _resolve_pai_save_name(args.output_dir, args.pai_save_name)
    )

    # Start clean on fresh runs so old converted modules do not leak into the next experiment.
    # In DDP, only rank 0 performs destructive filesystem cleanup.
    if not args.perforated_load_path:
        if is_main_process and os.path.exists(pai_system_name):
            print(f"Removing old PAI directory at {pai_system_name}...")
            try:
                shutil.rmtree(pai_system_name)
            except FileNotFoundError:
                # Another process or NFS delay may make entries disappear mid-delete.
                pass
        if is_distributed:
            dist.barrier()

    # Keep these directories available for tracker CSV export variants.
    # PerforatedAI can build both absolute and cwd-relative variants internally.
    for pai_root in _pai_path_variants(pai_system_name):
        os.makedirs(pai_root, exist_ok=True)
        os.makedirs(os.path.join(pai_root, args.output_dir.lstrip(os.sep)), exist_ok=True)
        os.makedirs(os.path.join(pai_root, args.output_dir), exist_ok=True)

    if args.perforated_load_path:
        model = UPA.initialize_pai(model, save_name=pai_system_name)
        if hasattr(UPA, "load_system"):
            try:
                model = UPA.load_system(model, pai_system_name, "latest", True)
            except BaseException as exc:
                print(f"PerforatedAI load_system failed: {exc}. Continuing with initialized model.")
    else:
        model = UPA.initialize_pai(model, save_name=pai_system_name)

    _assert_disallowed_module_not_converted(model, disallowed_module_id)

    print(model.classifier_fc)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    model = model.to(device)
    model_without_ddp = model
    optimizer, scheduler = setup_pai_optimizer_scheduler(args, model_without_ddp)

    # Wrap model with DistributedDataParallel if in distributed mode
    if is_distributed:
        # When using PAI dendrites, always enable find_unused_parameters because newly added
        # parameters may not participate in every forward pass during early training.
        find_unused = args.ddp_find_unused_parameters or (args.dendrite_mode > 0)
        model = DDP(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=find_unused,
            broadcast_buffers=False,
        )
        parallel_model = model
    else:
        parallel_model = model

    running_stats: Dict[str, float] = {}
    best_validation_accuracy = float("-inf")
    best_validation_snapshot: Dict[str, float] = {}
    start_epoch = 1
    warned_missing_pai_nonmain = False

    if args.resume_from_checkpoint and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Handle both wrapped (DDP) and unwrapped model state dicts
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            running_stats = checkpoint.get("running_stats", running_stats)
            best_validation_accuracy = checkpoint.get("best_validation_accuracy", best_validation_accuracy)
            best_validation_snapshot = checkpoint.get("best_validation_snapshot", best_validation_snapshot)
            loaded_epoch = int(checkpoint.get("epoch", 0))
            start_epoch = loaded_epoch + 1
            if is_main_process:
                print(f"Resuming from epoch {loaded_epoch}")
        except Exception as exc:
            if is_main_process:
                print(f"Failed to resume from checkpoint at {checkpoint_path}: {exc}")

    run = init_wandb(args) if is_main_process else None
    if is_distributed:
        dist.barrier()  # Synchronize all ranks after setup
    cycle_start = time.perf_counter()


    epoch = start_epoch - 1
    while True:
        epoch += 1
        epoch_start = time.perf_counter()

        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_accuracy, train_top5_accuracy = train(
            args, parallel_model, device, train_loader, optimizer, epoch, is_main_process=is_main_process
        )
        if is_main_process:
            GPA.pai_tracker.add_extra_score(train_accuracy, "train")
            GPA.pai_tracker.add_extra_score(train_top5_accuracy, "train_top5")
        val_metrics, test_metrics = test(
            parallel_model,
            device,
            val_loader,
            test_loader,
            is_main_process=is_main_process,
        )

        # For DDP, need to pass unwrapped model to PAI; then re-wrap
        model_for_pai = model.module if isinstance(model, DDP) else model
        rebuilt_switch_model = False
        if is_main_process:
            if not _model_has_pai_modules(model_for_pai):
                if is_main_process:
                    print("Detected model without PAI modules before switch logic; rebuilding PAI switch model.")
                model_for_pai = prepare_pai_switch_model(
                    model_for_pai,
                    pai_system_name,
                    args.finetune_backbone,
                    disallowed_module_id=disallowed_module_id,
                )
                rebuilt_switch_model = True
            _assert_disallowed_module_not_converted(model_for_pai, disallowed_module_id)
            # Rebind optimizer/scheduler only if we had to rebuild the wrapped model.
            if rebuilt_switch_model:
                optimizer, scheduler = setup_pai_optimizer_scheduler(args, model_for_pai)
        elif not _model_has_pai_modules(model_for_pai):
            # Non-main ranks do not run switch logic; avoid repeated recovery/load attempts
            # here because they can emit noisy false-positive load_system warnings.
            if not warned_missing_pai_nonmain:
                print(
                    "Non-main rank detected model without explicit PAI markers; "
                    "deferring recovery to rank-synchronized restructure path."
                )
                warned_missing_pai_nonmain = True

        # Seed switch-mode files once on rank 0, then synchronize ranks.
        ensure_pai_switch_files_exist(
            pai_system_name,
            model_for_pai,
            is_distributed=is_distributed,
            is_main_process=is_main_process,
        )
        restructured = False
        training_complete = False
        pai_switch_error = False
        if is_main_process:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="No artists with labels found to put in legend.*",
                        category=UserWarning,
                    )
                    model_for_pai, restructured, training_complete = _call_pai_add_validation_score(
                        args,
                        float(val_metrics["accuracy"]),
                        model_for_pai,
                    )
            except SystemExit as exc:
                if _is_nonfatal_pai_system_exit(exc):
                    print(
                        "PAI add_validation_score returned non-fatal SystemExit "
                        f"(code={exc.code!r}); continuing without restructure this epoch."
                    )
                else:
                    raise
            except Exception as exc:
                print(f"PAI add_validation_score failed once: {exc}. Attempting recovery and single retry.")
                try:
                    model_for_pai = recover_pai_model_if_needed(model_for_pai, pai_system_name)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="No artists with labels found to put in legend.*",
                            category=UserWarning,
                        )
                        model_for_pai, restructured, training_complete = _call_pai_add_validation_score(
                            args,
                            float(val_metrics["accuracy"]),
                            model_for_pai,
                        )
                    print("PAI add_validation_score retry succeeded after recovery.")
                except SystemExit as retry_exc:
                    if _is_nonfatal_pai_system_exit(retry_exc):
                        print(
                            "PAI add_validation_score retry returned non-fatal SystemExit "
                            f"(code={retry_exc.code!r}); continuing without restructure this epoch."
                        )
                    else:
                        raise
                except Exception as retry_exc:
                    pai_switch_error = True
                    training_complete = True
                    print(
                        "PAI add_validation_score retry failed; "
                        f"ending training cleanly to avoid DDP rank desync: {retry_exc}"
                    )

        if is_distributed:
            # Ensure main rank's PAI switch is fully persisted before non-main ranks try to load.
            dist.barrier()
            
            control = [bool(restructured), bool(training_complete), bool(pai_switch_error)]
            dist.broadcast_object_list(control, src=0)
            restructured = bool(control[0])
            training_complete = bool(control[1])
            pai_switch_error = bool(control[2])
            if restructured and hasattr(UPA, "load_system"):
                dist.barrier()  # Wait for main rank's save to persist
                latest_path = os.path.join(os.path.abspath(os.path.normpath(pai_system_name)), "latest.pt")
                if not _model_has_pai_modules(model_for_pai):
                    model_for_pai = UPA.initialize_pai(model_for_pai, save_name=pai_system_name)
                # Retry with NFS latency tolerance
                for attempt in range(5):
                    if os.path.exists(latest_path):
                        break
                    if attempt < 4:
                        time.sleep(0.1)
                
                if os.path.exists(latest_path):
                    try:
                        model_for_pai = UPA.load_system(model_for_pai, pai_system_name, "latest", True)
                        if is_main_process:
                            print(f"All ranks reloaded restructured model from {latest_path}")
                    except Exception as exc:
                        if is_main_process:
                            print(f"Warning: failed to load restructured model: {exc}")
            if pai_switch_error and not is_main_process:
                print("Rank 0 reported PAI switch failure; ending training cleanly on this rank.")

        model_for_pai = model_for_pai.to(device)

        if restructured:
            # Ensure all ranks have synchronized their model state before re-wrapping with DDP.
            if is_distributed:
                dist.barrier()
            
            # Rebuild DDP wrapper only when model structure changes.
            optimizer, scheduler = setup_pai_optimizer_scheduler(args, model_for_pai)
            if is_distributed:
                # When using PAI dendrites, always enable find_unused_parameters because newly added
                # parameters may not participate in every forward pass during early training.
                find_unused = args.ddp_find_unused_parameters or (args.dendrite_mode > 0)
                model = DDP(
                    model_for_pai,
                    device_ids=[args.gpu],
                    output_device=args.gpu,
                    find_unused_parameters=find_unused,
                    broadcast_buffers=False,
                )
            else:
                model = model_for_pai
            parallel_model = model
        else:
            # Keep existing wrapper to avoid rank desync on PAI internal variable-size buffers.
            if not is_distributed:
                model = model_for_pai
                parallel_model = model
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Detected call of `lr_scheduler\\.step\\(\\)` before `optimizer\\.step\\(\\)`.*",
                    category=UserWarning,
                )
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
            if is_main_process:
                try:
                    # Unwrap model if DDP before saving
                    model_to_save = model.module if isinstance(model, DDP) else model
                    save_on_master(
                        _state_dict_without_pai_metadata(
                            model_to_save,
                            disallowed_module_id=disallowed_module_id,
                        ),
                        best_model_path,
                    )
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

        if is_main_process:
            print(f"Epoch {epoch} metrics: {epoch_log}")
        if run is not None:
            log_to_wandb(run, epoch_log, step=epoch)

        # Always save checkpoint from main process; extract unwrapped state dict for DDP
        if is_main_process:
            checkpoint_model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": checkpoint_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "running_stats": running_stats,
                "best_validation_accuracy": best_validation_accuracy,
                "best_validation_snapshot": best_validation_snapshot,
                "finetune_backbone": args.finetune_backbone,
            }
            try:
                save_on_master(checkpoint, checkpoint_path)
            except Exception as exc:
                print(f"Failed to save checkpoint to {checkpoint_path}: {exc}")

        if training_complete:
            if is_main_process:
                print("PerforatedAI signaled training complete.")
            break

        if args.epochs > 0 and epoch >= args.epochs:
            if is_main_process:
                print(f"Reached --epochs {args.epochs} safety cap.")
            break

    if is_distributed:
        dist.barrier()

    # Unwrap model before final evaluation if using DDP
    if isinstance(model, DDP):
        model = model.module
    final_metrics: Dict[str, object] = {}
    if is_main_process:
        model.eval()

        if os.path.exists(best_model_path):
            try:
                _load_state_dict_compatible(
                    model,
                    best_model_path,
                    device,
                    disallowed_module_id=disallowed_module_id,
                )
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

        final_metrics = {
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

        pai_png_path = ensure_pai_png(args.output_dir, model, pai_system_name)
        print(f"PAI graph image written to: {pai_png_path}")
        if run is not None and os.path.exists(pai_png_path):
            try:
                run.log({"perforatedai/pai_graph": wandb.Image(pai_png_path)})
            except Exception as exc:
                print(f"W&B PAI.png upload failed: {exc}")

    if run is not None:
        log_to_wandb(run, final_metrics)
        finish_wandb(run)

    if args.save_model and is_main_process:
        save_on_master(model.state_dict(), os.path.join(args.output_dir, "efficientnet_b5_flowers102_baseline.pt"))
    
    # Clean up distributed training
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
