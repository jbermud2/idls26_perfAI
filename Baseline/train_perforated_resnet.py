"""
CUDA_VISIBLE_DEVICES=0 python train_perforated_resnet.py --model resnet18 --batch-size 32 --lr 0.0125 --val-resize-size 256 --val-crop-size 224 --train-crop-size 224 --data-path /ocean/projects/cis260045p/shared/data/imagenet --convert-count 0 --dendrite-mode 1 --improvement-threshold 1 --candidate-weight-init-mult 0.1 --pai-forward-function relu --use-wandb --wandb-project IMAGENET_PERFORATED --workers 10 --max-dendrites 3

usage:
CUDA_VISIBLE_DEVICES=0 python -m pdb train_fast_perforatedai_from_config.py --model resnet18 --batch-size 32 --lr 0.0125 --val-resize-size 256 --val-crop-size 224 --train-crop-size 224 --full-dataset --data-path /home/rbrenner/Datasets/imagenet --convert-count 0 --dendrite-mode 2 --improvement-threshold 1 --candidate-weight-init-mult 0.1 --pai-forward-function relu
"""


"""
CUDA_VISIBLE_DEVICES=1 python -m pdb train_fast_perforatedai_from_config.py --model resnet50 --batch-size 32 --lr 0.0125 --val-resize-size 256 --val-crop-size 224 --train-crop-size 224 --full-dataset --data-path /home/rbrenner/Datasets/imagenet --convert-count 0 --dendrite-mode 1 --improvement-threshold 1 --candidate-weight-init-mult 0.1 --pai-forward-function relu --perforated-load-path resnet50_c0_wd0.0001_dmode1_20260128_174922

"""

"""
Original:

Epoch: [89]  [500/506]  eta: 0:00:00  lr: 0.0010000000000000002  img/s: 12316.37788483597  loss: 0.5689 (0.5906)  acc1: 83.2031 (83.4479)  acc5: 95.7031 (95.5636)  time: 0.0253  data: 0.0049  max mem: 3617
Epoch: [89] Total time: 0:00:13
Test:   [ 0/20]  eta: 0:00:18  loss: 0.8479 (0.8479)  acc1: 74.6094 (74.6094)  acc5: 95.3125 (95.3125)  time: 0.9173  data: 0.9109  max mem: 3617
Test:  Total time: 0:00:01
Test:  Acc@1 70.300 Acc@5 89.820
Training time 0:22:53

with PAI:

Epoch: [1125]  [500/506]  eta: 0:00:01  lr: 0.00010000000000000003  img/s: 1421.8356206467333  loss: 0.3493 (0.3511)  acc1: 90.2344 (90.1432)  acc5: 96.8750 (97.4715)  time: 0.1801  data: 0.0001  max mem: 9340
Epoch: [1125] Total time: 0:01:32
Adding extra score Train Acc 1 of 90.16036168026996
Adding extra score Train Acc 5 of 97.47517291776981
Test:   [ 0/20]  eta: 0:00:22  loss: 0.6772 (0.6772)  acc1: 85.1562 (85.1562)  acc5: 96.8750 (96.8750)  time: 1.1152  data: 1.0578  max mem: 9340
Test:  Total time: 0:00:02
Test:  Acc@1 73.040 Acc@5 91.420


18_thin
Test:  Acc@1 61.660 Acc@5 85.200

current command, maybe working?:
CUDA_VISIBLE_DEVICES=0 python train_fast_perforatedai_from_config.py     --model resnet18     --batch-size 32     --lr 0.0125     --val-resize-size 256     --val-crop-size 224     --train-crop-size 224     --full-dataset     --data-path /home/rbrenner/Datasets/imagenet     --convert-count 0     --dendrite-mode 1     --improvement-threshold 0     --candidate-weight-init-mult 0.1     --pai-forward-function relu --perforated-load-path resnet18_c0_wd0.0001_dmode1_20260112_205006



"""



import datetime
import math
import os
import time
import warnings
import argparse

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Import custom ResNet models
import resnet_double as custom_resnet

try:
    import wandb
except ImportError:
    wandb = None

from types import SimpleNamespace


# ---------------------------------------------------------------------------
# W&B sweep configuration
# ---------------------------------------------------------------------------

sweep_config = {"method": "random"}
sweep_config["metric"] = {"name": "ValAcc", "goal": "maximize"}

parameters_dict = {
    "dropout":                        {"values": [0.0, 0.1, 0.3, 0.5]},
    "weight_decay":                   {"values": [0, 1e-4]},
    "improvement_threshold":          {"values": [0, 1, 2]},
    "candidate_weight_init_mult":     {"values": [0.1, 0.01]},
    "pai_forward_function":           {"values": ["sigmoid", "relu", "tanh"]},
    "dendrite_graph_mode":            {"values": [True, False]},
    "dendrite_learn_mode":            {"values": [True, False]},
}


# ---------------------------------------------------------------------------
# W&B helper functions (modelled after mnist_perf.py)
# ---------------------------------------------------------------------------

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

    api_key = os.environ.get("WANDB_API_KEY", "") or getattr(args, "wandb_api_key", "")

    if args.wandb_mode == "online" and not api_key and getattr(args, "wandb_anonymous", "never") == "never":
        print("W&B online mode needs an API key. Set WANDB_API_KEY or pass --wandb-api-key.")
        print("Skipping W&B init to avoid interactive login prompt.")
        return None
    if args.wandb_mode == "offline":
        print("W&B running in offline mode. Use 'wandb sync' later to upload.")

    try:
        if api_key:
            wandb.login(key=api_key)
        elif args.wandb_mode == "online":
            print("W&B has no API key. Online runs may fail unless this machine is already logged in.")
    except Exception as exc:
        print(f"W&B login failed: {exc}")

    run_config = vars(args).copy()
    run_config.pop("wandb_api_key", None)

    init_kwargs = {
        "project":   args.wandb_project,
        "entity":    args.wandb_entity if getattr(args, "wandb_entity", "") else None,
        "name":      args.wandb_run_name if getattr(args, "wandb_run_name", "") else None,
        "mode":      args.wandb_mode,
        "anonymous": getattr(args, "wandb_anonymous", "never"),
        "id":        args.wandb_run_id if getattr(args, "wandb_run_id", "") else None,
        "resume":    args.wandb_resume if getattr(args, "wandb_run_id", "") else None,
        "config":    run_config,
    }

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"W&B init failed: {exc}")
        if init_kwargs.get("entity") is not None:
            print("Retrying without explicit entity.")
            init_kwargs["entity"] = None
            try:
                run = wandb.init(**init_kwargs)
            except Exception as retry_exc:
                print(f"W&B init retry failed: {retry_exc}")
                return None
        else:
            return None

    print(f"W&B initialized: project={args.wandb_project}, mode={args.wandb_mode}, run id={run.id}")
    return run


def log_to_wandb(run, metrics, step=None):
    if run is None:
        return
    try:
        if step is not None:
            run.log(metrics, step=step)
        else:
            run.log(metrics)
    except Exception as exc:
        print(f"W&B log failed: {exc}")


def finish_wandb(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"W&B finish failed: {exc}")


def safe_number(value):
    import math
    try:
        if value is None:
            return None
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return value
    except (TypeError, ValueError):
        return None


def update_min_max(stats, key, value):
    value = safe_number(value)
    if value is None:
        return
    stats[f"{key}_min"] = min(stats.get(f"{key}_min", value), value)
    stats[f"{key}_max"] = max(stats.get(f"{key}_max", value), value)


def benchmark_inference_throughput(model, data_loader, device, max_batches=20):
    model_was_training = model.training
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


def benchmark_latency_ms(model, data_loader, device, max_batches=20):
    model_was_training = model.training
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


def benchmark_cpu_latency_single_core_ms(model, data_loader, max_batches=20):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return benchmark_latency_ms(model, data_loader, torch.device("cpu"), max_batches=max_batches)
    finally:
        torch.set_num_threads(previous_threads)


def estimate_flops_with_hooks(model, device, input_shape=(1, 3, 224, 224)):
    total_flops = 0.0
    handles = []
    model_was_training = model.training

    def conv_hook(module, inputs, output):
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

    def linear_hook(module, inputs, output):
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
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)

    for handle in handles:
        handle.remove()

    if model_was_training:
        model.train()

    return float(total_flops)


def compute_model_stats(model, device, input_shape=(1, 3, 224, 224)):
    param_count = sum(p.numel() for p in model.parameters())
    flops = float("nan")
    flops_source = "approximate_hooks"

    try:
        flops = estimate_flops_with_hooks(model, device, input_shape=input_shape)
    except Exception as exc:
        print(f"FLOPS estimation failed: {exc}")
        flops = float("nan")

    return param_count, flops, flops_source


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    
    # Add training accuracies to PerforatedAI tracker
    GPA.pai_tracker.add_extra_score(metric_logger.acc1.global_avg, "Train Acc 1")
    GPA.pai_tracker.add_extra_score(metric_logger.acc5.global_avg, "Train Acc 5")


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    
    # gather the stats from all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")

    # Add validation score to PerforatedAI tracker and check for restructuring
    GPA.pai_tracker.add_extra_score(metric_logger.acc5.global_avg, "Val Acc 5")
    model, restructured, trainingComplete = GPA.pai_tracker.add_validation_score(metric_logger.acc1.global_avg, model)
    
    return model, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, metric_logger.loss.global_avg, restructured, trainingComplete


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


# ImageNet-100 standard class indices (commonly used subset)
IMAGENET100_CLASSES = [
    'n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475',
    'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878',
    'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544',
    'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084',
    'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779',
    'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777',
    'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541',
    'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366',
    'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811',
    'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457',
    'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322',
    'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381',
    'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939',
    'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244',
    'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797',
    'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675',
    'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143',
    'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313',
    'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805',
    'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672'
]


def filter_imagenet100(dataset):
    """Filter dataset to only include ImageNet-100 classes."""
    # Get original class_to_idx mapping
    original_class_to_idx = dataset.class_to_idx
    
    # Create mapping from old indices to new indices
    valid_classes = [cls for cls in IMAGENET100_CLASSES if cls in original_class_to_idx]
    new_class_to_idx = {cls: new_idx for new_idx, cls in enumerate(valid_classes)}
    old_to_new_idx = {original_class_to_idx[cls]: new_idx for cls, new_idx in new_class_to_idx.items()}
    
    # Filter samples
    filtered_samples = []
    for path, old_idx in dataset.samples:
        if old_idx in old_to_new_idx:
            filtered_samples.append((path, old_to_new_idx[old_idx]))
    
    # Update dataset
    dataset.samples = filtered_samples
    dataset.targets = [s[1] for s in filtered_samples]
    dataset.classes = valid_classes
    dataset.class_to_idx = new_class_to_idx
    
    print(f"Filtered dataset to {len(valid_classes)} classes with {len(filtered_samples)} samples")
    return dataset


def create_optimizer_and_scheduler(model, args, custom_keys_weight_decay, epoch=None):
    """Create optimizer and scheduler for the model using PerforatedAI setup.
    
    Args:
        model: The model to create optimizer for
        args: Training arguments
        custom_keys_weight_decay: List of (key, weight_decay) tuples for custom weight decay
        epoch: Current epoch (used for warmup adjustment after restructuring), None for initial setup
    
    Returns:
        optimizer, lr_scheduler tuple
    """
    # Set up parameter groups with different weight decay
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    # Set optimizer class
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        GPA.pai_tracker.set_optimizer(torch.optim.SGD)
        optimArgs = {
            "params": parameters,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "nesterov": "nesterov" in opt_name,
        }
    elif opt_name == "rmsprop":
        GPA.pai_tracker.set_optimizer(torch.optim.RMSprop)
        optimArgs = {
            "params": parameters,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "eps": 0.0316,
            "alpha": 0.9,
        }
    elif opt_name == "adamw":
        GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
        optimArgs = {
            "params": parameters,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")


    # Set scheduler class and prepare scheduler args
    args.lr_scheduler = args.lr_scheduler.lower()
    warmup_epochs_remaining = args.lr_warmup_epochs if epoch is None else max(0, args.lr_warmup_epochs - epoch)
    
    # Prepare main scheduler args
    if args.lr_scheduler == "steplr":
        main_schedArgs = {
            "step_size": args.lr_step_size,
            "gamma": args.lr_gamma,
        }
    elif args.lr_scheduler == "cosineannealinglr":
        main_schedArgs = {
            "T_max": args.epochs - args.lr_warmup_epochs,
            "eta_min": args.lr_min,
        }
    elif args.lr_scheduler == "exponentiallr":
        main_schedArgs = {
            "gamma": args.lr_gamma,
        }
    elif args.lr_scheduler == "reducelronplateau":
        main_schedArgs = {
            "mode": "max",
            "factor": 0.1,
            "patience": 10,
        }
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR, ExponentialLR and ReduceLROnPlateau "
            "are supported."
        )

    # If warmup is needed, create main scheduler manually and wrap with warmup using SequentialLR
    # Note: ReduceLROnPlateau cannot be used with SequentialLR, so skip warmup for it
    if warmup_epochs_remaining > 0 and args.lr_scheduler != "reducelronplateau":
        # Set scheduler to SequentialLR for PerforatedAI
        GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.SequentialLR)
        
        # Determine main scheduler class
        if args.lr_scheduler == "steplr":
            main_scheduler_class = torch.optim.lr_scheduler.StepLR
        elif args.lr_scheduler == "cosineannealinglr":
            main_scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
        elif args.lr_scheduler == "exponentiallr":
            main_scheduler_class = torch.optim.lr_scheduler.ExponentialLR
        
        # Determine warmup scheduler class and args
        if args.lr_warmup_method == "linear":
            warmup_scheduler_class = torch.optim.lr_scheduler.LinearLR
            warmup_schedArgs = {
                "start_factor": args.lr_warmup_decay,
                "total_iters": warmup_epochs_remaining,
            }
        elif args.lr_warmup_method == "constant":
            warmup_scheduler_class = torch.optim.lr_scheduler.ConstantLR
            warmup_schedArgs = {
                "factor": args.lr_warmup_decay,
                "total_iters": warmup_epochs_remaining,
            }
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        
        # Create SequentialLR args with both scheduler classes and their kwargs
        sequential_schedArgs = {
            "schedulers": [
                (warmup_scheduler_class, warmup_schedArgs),
                (main_scheduler_class, main_schedArgs)
            ],
            "milestones": [warmup_epochs_remaining]
        }
        optimizer, lr_scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, sequential_schedArgs)
    else:
        # No warmup needed, just create optimizer and scheduler through PerforatedAI
        if args.lr_scheduler == "steplr":
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.StepLR)
        elif args.lr_scheduler == "cosineannealinglr":
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR)
        elif args.lr_scheduler == "exponentiallr":
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ExponentialLR)
        elif args.lr_scheduler == "reducelronplateau":
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        optimizer, lr_scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, main_schedArgs)
        
        
        
    return optimizer, lr_scheduler


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
        )
        # Filter to ImageNet-100 unless full dataset is requested
        if not args.full_dataset:
            dataset = filter_imagenet100(dataset)
        
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        # Filter to ImageNet-100 unless full dataset is requested
        if not args.full_dataset:
            dataset_test = filter_imagenet100(dataset_test)
        
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args, wandb_run=None):
    # Use a pre-initialised sweep run if supplied, otherwise init normally.
    run = wandb_run if wandb_run is not None else init_wandb(args)
    if run is not None:
        print(f"Logging to wandb run: {run.name}")
    
    print(f"Config: model={args.model}, convert_count={args.convert_count}, weight_decay={args.weight_decay}")
    print(f"LR config: scheduler={args.lr_scheduler}, warmup_epochs={args.lr_warmup_epochs}, warmup_method={args.lr_warmup_method}")
    print(f"Aug config: label_smooth={args.label_smoothing}, mixup={args.mixup_alpha}, cutmix={args.cutmix_alpha}, "
          f"random_erase={args.random_erase}, dropout={args.dropout}, auto_aug={args.auto_augment}")
    print(f"PAI config: improvement_threshold={args.improvement_threshold}, "
          f"init_mult={args.candidate_weight_init_mult}, "
          f"forward_fn={args.pai_forward_function}, dendrite_mode={args.dendrite_mode}")
    
    if args.output_dir:
        utils.mkdir(args.output_dir)

    # Apply batch_lr_factor scaling
    if args.batch_lr_factor != 1.0:
        original_batch_size = args.batch_size
        original_lr = args.lr
        args.batch_size = int(args.batch_size * args.batch_lr_factor)
        args.lr = args.lr * args.batch_lr_factor
        print(f"Applied batch_lr_factor={args.batch_lr_factor}: batch_size {original_batch_size}->{args.batch_size}, lr {original_lr}->{args.lr}")

    args.distributed = False

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = "/ocean/projects/cis260045p/shared/data/imagenet/train/train_blurred"
    val_dir = "/ocean/projects/cis260045p/shared/data/imagenet/val/val_blurred"
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    num_classes = len(dataset.classes)
    dataset_type = "full ImageNet" if args.full_dataset else "ImageNet-100 subset"
    print(f"Training with {num_classes} classes ({dataset_type})")
    
    # Set up PerforatedAI global parameters
    GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_n_epochs_to_switch(40)
    GPA.pc.set_p_epochs_to_switch(40)
    GPA.pc.set_cap_at_n(True)
    GPA.pc.set_initial_history_after_switches(2)
    GPA.pc.set_test_saves(True)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.append_module_names_to_convert(["BasicBlock", "Bottleneck"])
    GPA.pc.set_verbose(False)
    #GPA.pc.set_max_dendrites(3)
    
    # Apply PAI settings from command-line args
    if args.improvement_threshold == 0:
        thresh = [0.01, 0.001, 0.0001, 0]
    elif args.improvement_threshold == 1:
        thresh = [0.001, 0.0001, 0]
    elif args.improvement_threshold == 2:
        thresh = [0]
    GPA.pc.set_improvement_threshold(thresh)
    
    GPA.pc.set_candidate_weight_initialization_multiplier(args.candidate_weight_init_mult)
    
    # Decode pai_forward_function from string
    if args.pai_forward_function == "sigmoid":
        pai_forward_function = torch.sigmoid
    elif args.pai_forward_function == "relu":
        pai_forward_function = torch.relu
    elif args.pai_forward_function == "tanh":
        pai_forward_function = torch.tanh
    else:
        pai_forward_function = torch.sigmoid
    GPA.pc.set_pai_forward_function(pai_forward_function)
    
    # Set dendrite mode
    if args.dendrite_mode == 0:
        GPA.pc.set_max_dendrites(0)
    elif args.dendrite_mode == 1:
        GPA.pc.set_max_dendrites(args.max_dendrites)
        GPA.pc.set_perforated_backpropagation(False)
    elif args.dendrite_mode == 2:
        GPA.pc.set_max_dendrites(args.max_dendrites)
        GPA.pc.set_perforated_backpropagation(True)
    
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )
    # Match mnist_perf.py metric setup: CPU batch=1, GPU batch=100 for inference throughput.
    cpu_benchmark_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )
    gpu_benchmark_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=100, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    # Check if it's one of our custom models
    if args.model in ['resnet18_thin', 'resnet10_shallow', 'resnet12_balanced']:
        model_fn = getattr(custom_resnet, args.model)
        model = model_fn(num_classes=num_classes)
        print(f"Created custom model: {args.model}")
    else:
        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    
    # Apply dropout if specified (add dropout after global average pooling, before final classifier)
    if args.dropout > 0.0:
        # For ResNet models, insert dropout before the final fc layer
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            fc = nn.Sequential(
                nn.Dropout(p=args.dropout),
                nn.Linear(in_features, num_classes)
            )
            fc.in_features = in_features
            model.fc = fc
            print(f"Applied dropout rate: {args.dropout}")
    
    # Apply stochastic depth if specified (for ResNet models)
    if args.stochastic_depth_prob > 0.0:
        print(f"Note: Stochastic depth rate {args.stochastic_depth_prob} specified, but requires model recreation with stochastic_depth parameter")
        print(f"Consider using: torchvision.models.resnet18(weights=None, num_classes={num_classes}, stochastic_depth_prob={args.stochastic_depth_prob})")
    
    # Note on width/depth multipliers
    if args.width_multiplier != 1.0 or args.depth_multiplier != 1.0:
        print(f"Note: Width multiplier {args.width_multiplier} and/or depth multiplier {args.depth_multiplier} specified")
        print(f"These require custom model creation. Consider using smaller models like resnet18 or using torchvision.models.efficientnet with different variants")
    

    skip_layers = 4-args.convert_count

    for i in range(skip_layers):
        GPA.pc.append_module_ids_to_track(['.layer'+str(i+1)])
    GPA.pc.append_module_ids_to_track(['.conv1', '.b1', '.fc'])
    # Wrap model with PerforatedAI
    model = custom_resnet.ResNetPAI(model)
    # Build save name
    save_name = f"{args.model}_c{args.convert_count}_wd{args.weight_decay}_dmode{args.dendrite_mode}"
    if run is not None:
        run.name = save_name
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name_with_timestamp = f"{save_name}_{timestamp}"
    
    # Load from checkpoint if path provided, otherwise initialize new
    if args.perforated_load_path != '':
        model = UPA.initialize_pai(model, save_name=args.perforated_load_path)
        model = UPA.load_system(model, args.perforated_load_path, 'latest', True)
    else:
        model = UPA.initialize_pai(model, save_name=save_name_with_timestamp)

    #import pdb; pdb.set_trace()

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    
    # Create optimizer and scheduler
    optimizer, lr_scheduler = create_optimizer_and_scheduler(model, args, custom_keys_weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    cycle_start = time.perf_counter()
    epoch = args.start_epoch - 1

    # -----------------------------------------------------------------------
    # Tracking variables (mirrors mnist_perf.py pattern)
    # -----------------------------------------------------------------------
    running_stats = {}
    best_validation_accuracy = float("-inf")
    best_validation_snapshot = {}

    # Per-architecture max tracking (for Arch Max logging on dendrite add)
    arch_max_val   = 0.0
    arch_max_train = 0.0
    arch_max_params = 0
    dendrite_count = 0

    # Global max across all architectures
    global_max_val   = 0.0
    global_max_train = 0.0
    global_max_params = 0

    while True:
        epoch += 1
        epoch_start = time.perf_counter()

        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)

        model, acc1, acc5, val_loss, restructured, trainingComplete = evaluate(model, criterion, data_loader_test, device=device)

        seconds_per_training_epoch  = time.perf_counter() - epoch_start
        seconds_per_training_cycle  = time.perf_counter() - cycle_start

        # Pull training accuracy stored by train_one_epoch via add_extra_score
        train_acc1 = GPA.pai_tracker.member_vars.get("extra_scores", {}).get("Train Acc 1", 0.0)
        train_acc5 = GPA.pai_tracker.member_vars.get("extra_scores", {}).get("Train Acc 5", 0.0)
        param_count  = UPA.count_params(model)
        num_dendrites = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)

        # ---- running min/max -----------------------------------------------
        update_min_max(running_stats, "val_acc1",   acc1)
        update_min_max(running_stats, "val_acc5",   acc5)
        update_min_max(running_stats, "train_acc1", train_acc1)
        update_min_max(running_stats, "train_acc5", train_acc5)

        # ---- best-validation snapshot --------------------------------------
        if acc1 > best_validation_accuracy:
            best_validation_accuracy = acc1
            best_validation_snapshot = {
                "val_acc1_best":             acc1,
                "val_acc5_at_best_val":      acc5,
                "train_acc1_at_best_val":    train_acc1,
                "train_acc5_at_best_val":    train_acc5,
                "param_count_at_best_val":   param_count,
                "epoch_at_best_val":         epoch,
            }

        # ---- per-architecture max ------------------------------------------
        if acc1 > arch_max_val:
            arch_max_val   = acc1
            arch_max_train = train_acc1
            arch_max_params = param_count
        if acc1 > global_max_val:
            global_max_val   = acc1
            global_max_train = train_acc1
            global_max_params = param_count

        # ---- per-epoch W&B log (matches mnist_perf metric names) -----------
        epoch_log = {
            "epoch":                                          epoch,
            # Core accuracy
            "ValAcc":                                         acc1,
            "TrainAcc":                                       train_acc1,
            # Top-5
            "perforatedai/val_acc5":                          acc5,
            "perforatedai/train_acc5":                        train_acc5,
            # Validation loss
            "perforatedai/val_loss":                          val_loss,
            # Precision@1 (== top-1 accuracy for single-label)
            "perforatedai/precision_at_1":                    acc1,
            # Running min/max
            "perforatedai/val_acc1_min":  running_stats.get("val_acc1_min"),
            "perforatedai/val_acc1_max":  running_stats.get("val_acc1_max"),
            "perforatedai/val_acc5_min":  running_stats.get("val_acc5_min"),
            "perforatedai/val_acc5_max":  running_stats.get("val_acc5_max"),
            "perforatedai/train_acc1_min": running_stats.get("train_acc1_min"),
            "perforatedai/train_acc1_max": running_stats.get("train_acc1_max"),
            "perforatedai/train_acc5_min": running_stats.get("train_acc5_min"),
            "perforatedai/train_acc5_max": running_stats.get("train_acc5_max"),
            # Timing (matches mnist_perf exactly)
            "perforatedai/seconds_per_training_epoch":        seconds_per_training_epoch,
            "perforatedai/seconds_per_training_cycle":        seconds_per_training_cycle,
            # Model size
            "Param Count":                                    param_count,
            "Dendrite Count":                                 num_dendrites,
        }

        if best_validation_snapshot:
            epoch_log["perforatedai/val_acc1_best"]            = best_validation_snapshot["val_acc1_best"]
            epoch_log["perforatedai/val_acc5_at_best_val"]     = best_validation_snapshot["val_acc5_at_best_val"]
            epoch_log["perforatedai/train_acc1_at_best_val"]   = best_validation_snapshot["train_acc1_at_best_val"]
            epoch_log["perforatedai/train_acc5_at_best_val"]   = best_validation_snapshot["train_acc5_at_best_val"]
            epoch_log["perforatedai/param_count_at_best_val"]  = best_validation_snapshot["param_count_at_best_val"]
            epoch_log["perforatedai/epoch_at_best_val"]        = best_validation_snapshot["epoch_at_best_val"]

        log_to_wandb(run, epoch_log, step=epoch)

        # ---- architecture-max log (when a new dendrite batch is added) -----
        if run is not None and restructured:
            if (GPA.pai_tracker.member_vars.get("mode") == "n" and
                    dendrite_count != num_dendrites):
                dendrite_count = num_dendrites
                log_to_wandb(run, {
                    "Arch Max Val":        arch_max_val,
                    "Arch Max Train":      arch_max_train,
                    "Arch Param Count":    arch_max_params,
                    "Arch Dendrite Count": dendrite_count - 1,
                })
                # Reset per-arch trackers for next architecture
                arch_max_val   = 0.0
                arch_max_train = 0.0
                arch_max_params = 0

        # ---- reset optimizer/scheduler after restructuring -----------------
        if restructured:
            model.to(device)
            optimizer, lr_scheduler = create_optimizer_and_scheduler(model, args, custom_keys_weight_decay, epoch=epoch)

        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")

        if args.output_dir:
            checkpoint = {
                "model":        model_without_ddp.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch":        epoch,
                "args":         args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        # ---- training complete ---------------------------------------------
        if trainingComplete:
            print("PerforatedAI training complete!")
            final_metrics = {
                "Final Max Val":       global_max_val,
                "Final Max Train":     global_max_train,
                "Final Param Count":   global_max_params,
                "Final Dendrite Count": num_dendrites,
            }
            if best_validation_snapshot:
                final_metrics["perforatedai/val_acc1_best"]           = best_validation_snapshot["val_acc1_best"]
                final_metrics["perforatedai/val_acc5_at_best_val"]    = best_validation_snapshot["val_acc5_at_best_val"]
                final_metrics["perforatedai/train_acc1_at_best_val"]  = best_validation_snapshot["train_acc1_at_best_val"]
                final_metrics["perforatedai/train_acc5_at_best_val"]  = best_validation_snapshot["train_acc5_at_best_val"]
                final_metrics["perforatedai/param_count_at_best_val"] = best_validation_snapshot["param_count_at_best_val"]
                final_metrics["perforatedai/epoch_at_best_val"]       = best_validation_snapshot["epoch_at_best_val"]
            log_to_wandb(run, final_metrics)
            break

    print("Final Param Count:", UPA.count_params(model))
    model.eval()

    gpu_inference_ips = float("nan")
    if torch.cuda.is_available():
        gpu_inference_ips = benchmark_inference_throughput(model, gpu_benchmark_loader, torch.device("cuda"))

    model_cpu = model.to(torch.device("cpu"))
    cpu_inference_ips = benchmark_inference_throughput(model_cpu, cpu_benchmark_loader, torch.device("cpu"))
    latency_ms = benchmark_cpu_latency_single_core_ms(model_cpu, cpu_benchmark_loader)
    input_side = int(args.val_crop_size)
    param_count, flops, flops_source = compute_model_stats(
        model_cpu,
        torch.device("cpu"),
        input_shape=(1, 3, input_side, input_side),
    )

    final_benchmark_metrics = {
        "perforatedai/gpu_inference_inputs_per_second": safe_number(gpu_inference_ips),
        "perforatedai/cpu_inference_inputs_per_second": safe_number(cpu_inference_ips),
        "efficientnet/num_parameters": param_count,
        "efficientnet/flops": safe_number(flops),
        "efficientnet/flops_source": flops_source,
        "efficientnet/latency_ms_per_batch": safe_number(latency_ms),
    }
    if best_validation_snapshot:
        final_benchmark_metrics["efficientnet/accuracy_vs_flops"] = best_validation_snapshot["val_acc1_best"]
    log_to_wandb(run, final_benchmark_metrics)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    finish_wandb(run)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training with PerforatedAI (Fast - ImageNet-100, Half Resolution)", add_help=add_help)

    parser.add_argument("--data-path", default="/home/rbrenner/Datasets/imagenet", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=256, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "--batch-lr-factor", default=1.0, type=float, help="factor to scale batch size and learning rate (e.g., 0.5 halves batch size and scales lr accordingly)"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=None, type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=lambda x: None if x == 'None' else x, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Regularization parameters to reduce overfitting (train-val gap)
    parser.add_argument("--dropout", default=0.0, type=float, help="dropout rate (default: 0.0, no dropout)")
    parser.add_argument("--width-multiplier", default=1.0, type=float, help="network width multiplier to reduce capacity (default: 1.0, full width)")
    parser.add_argument("--depth-multiplier", default=1.0, type=float, help="network depth multiplier to reduce capacity (default: 1.0, full depth)")
    parser.add_argument(
        "--stochastic-depth-prob",
        default=0.0,
        type=float,
        help="stochastic depth drop probability for ResNet (default: 0.0, no stochastic depth)",
    )

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    # Half resolution defaults (128 instead of 256, 112 instead of 224)
    parser.add_argument(
        "--val-resize-size", default=128, type=int, help="the resize size used for validation (default: 128 for fast training)"
    )
    parser.add_argument(
        "--val-crop-size", default=112, type=int, help="the central crop size used for validation (default: 112 for fast training)"
    )
    parser.add_argument(
        "--train-crop-size", default=112, type=int, help="the random crop size used for training (default: 112 for fast training)"
    )
    parser.add_argument("--convert-count", default=0, type=int, help="total number of layers to convert")
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument("--full-dataset", action="store_true", help="Use full ImageNet-1000 instead of ImageNet-100 subset")
    
    # PerforatedAI parameters
    parser.add_argument("--improvement-threshold", default=0, type=int, choices=[0, 1, 2], 
                        help="PAI improvement threshold mode: 0=[0.01,0.001,0.0001,0], 1=[0.001,0.0001,0], 2=[0]")
    parser.add_argument("--candidate-weight-init-mult", default=0.1, type=float,
                        help="PAI candidate weight initialization multiplier (default: 0.1)")
    parser.add_argument("--pai-forward-function", default="sigmoid", type=str, choices=["sigmoid", "relu", "tanh"],
                        help="PAI forward function (default: sigmoid)")
    parser.add_argument("--dendrite-mode", default=1, type=int, choices=[0, 1, 2],
                        help="Dendrite mode: 0=no dendrites, 1=GD dendrites, 2=PB dendrites (default: 2)")
    parser.add_argument("--max-dendrites", default=5, type=int,
                        help="Maximum number of dendrites to add per layer (default: 5). No effect when --dendrite-mode 0.")
    parser.add_argument("--perforated-load-path", default="", type=str,
                        help="Path to load PerforatedAI checkpoint from (default: '', initialize new)")

    # Wandb logging
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="IMAGENET_PERFORATED",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="",
                        help="W&B entity (team) name (optional)")
    parser.add_argument("--wandb-run-name", type=str, default="",
                        help="W&B run name (optional, auto-generated if empty)")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="W&B mode (default: online)")
    parser.add_argument("--wandb-anonymous", type=str, default="never",
                        choices=["never", "allow", "must"],
                        help="W&B anonymous mode (default: never)")
    parser.add_argument("--wandb-run-id", type=str, default="",
                        help="W&B run id for resuming an existing run")
    parser.add_argument("--wandb-resume", type=str, default="allow",
                        choices=["allow", "must", "never"],
                        help="W&B resume behaviour when --wandb-run-id is provided")
    parser.add_argument("--wandb-api-key", type=str, default="",
                        help="W&B API key (preferred: set WANDB_API_KEY env var)")
    parser.add_argument("--sweep-id", type=str, default="main",
                        help="W&B sweep id to join. Use 'main' (default) to create a new sweep. "
                             "Pass the printed sweep id on additional machines to join as extra agents.")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    if not args.use_wandb:
        main(args)
    else:
        if wandb is None:
            raise ImportError("wandb is required. Install with: pip install wandb")

        sweep_config["parameters"] = parameters_dict
        project = args.wandb_project

        def run():
            try:
                with wandb.init(config=sweep_config) as wandb_run:
                    cfg = wandb_run.config
                    if hasattr(cfg, "dropout"):
                        args.dropout = cfg.dropout
                    if hasattr(cfg, "weight_decay"):
                        args.weight_decay = cfg.weight_decay
                    if hasattr(cfg, "improvement_threshold"):
                        args.improvement_threshold = cfg.improvement_threshold
                    if hasattr(cfg, "candidate_weight_init_mult"):
                        args.candidate_weight_init_mult = cfg.candidate_weight_init_mult
                    if hasattr(cfg, "pai_forward_function"):
                        args.pai_forward_function = cfg.pai_forward_function
                    if hasattr(cfg, "dendrite_graph_mode"):
                        GPA.pc.set_dendrite_graph_mode(cfg.dendrite_graph_mode)
                    if hasattr(cfg, "dendrite_learn_mode"):
                        GPA.pc.set_dendrite_update_mode(cfg.dendrite_learn_mode)

                    priorities = ["dendrite_mode", "model"]
                    excluded   = ["method", "metric", "parameters"]
                    name_parts = [str(wandb.config[k]) for k in priorities if k in wandb.config]
                    remaining  = [k for k in parameters_dict if k not in excluded and k not in priorities]
                    name_parts.extend(str(wandb.config[k]) for k in remaining if k in wandb.config)
                    wandb_run.name = "_".join(name_parts) if name_parts else wandb_run.name

                    main(args, wandb_run=wandb_run)
            except Exception:
                import pdb
                pdb.post_mortem()

        if args.sweep_id == "main":
            sweep_id = wandb.sweep(sweep_config, project=project)
            print(f"\nInitialized sweep. Use --sweep-id {sweep_id} to join on other machines.\n")
            wandb.agent(sweep_id, run, count=300)
        else:
            wandb.agent(args.sweep_id, run, count=300, project=project)
