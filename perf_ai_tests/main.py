'''
For running:
interact -p GPU-shared --gres=gpu:v100-32:1 -t 8:00:00 -A cis260045p

interact -p GPU-shared --gres=gpu:h100-80:1 -t 8:00:00 -A cis260045p

python -m torch.distributed.run --nproc_per_node 2 main.py --data-root /ocean/projects/cis260045p/shared/data --use-wandb --wandb-api-key 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 --dendrite-mode 1 --max-dendrites 5 --pai-forward-function relu --improvement-threshold 0.5 --candidate-weight-init-mult 0.1 --epochs 40 --batch-size 256 > output.txt 2>&1

python main.py --data-root /ocean/projects/cis260045p/shared/data --use-wandb --wandb-api-key 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 --dendrite-mode 1 --max-dendrites 4 --pai-forward-function relu --improvement-threshold 1 --candidate-weight-init-mult 0.1 --model efficientnet_b4 > output.txt 2>&1

'''

from __future__ import annotations

import argparse
import os
import time
import shutil
from typing import Dict
import json

import torch

from data.registry import get_dataset_builder
from models import EfficientNetPAI, NUM_CLASSES
from models.registry import get_model_build_config, list_model_build_config_names
from trainer.eval import test
from trainer.train import train
from utils.generic_utils import (
    set_seed,
    disable_interactive_breakpoints,
    cuda_gc
)
from utils.metrics_utils import (
    benchmark_cpu_latency_single_core_ms,
    benchmark_inference_throughput,
    compute_model_stats,
    count_parameters,
    refresh_flops_for_training,
    safe_number,
    update_min_max,
)
from utils.pai_utils import (
    call_pai_add_validation_score,
    configure_pai,
    selected_convert_module_id,
    setup_pai_optimizer_scheduler
)
from utils.wandb_utils import (
    finish_wandb, 
    init_wandb, 
    log_checkpoint_artifact,
    log_to_wandb
)

try:
    import wandb
except ImportError:
    wandb = None

try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
except ImportError:
    GPA = None
    UPA = None


FLOWERS_DATASET_ROOT_DEFAULT = "/ocean/projects/cis260045p/shared/data"
DEFAULT_PAI_CONVERT_TARGET = "pre_fc"
DATASET_REGISTRY_NAME = "flowers102"
DEFAULT_MODEL_NAME = "efficientnet_b4"


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, choices=list_model_build_config_names())
    pre_args, _ = pre_parser.parse_known_args()
    pre_model_config = get_model_build_config(pre_args.model)
    model_defaults = dict(pre_model_config.pai_arg_defaults)

    parser = argparse.ArgumentParser(
        description="PyTorch transfer learning with EfficientNet-B5 + PerforatedAI (No DDP)."
    )
    parser.add_argument("--batch-size", type=int, default=model_defaults.get("batch_size", 16), metavar="N")
    parser.add_argument("--test-batch-size", type=int, default=model_defaults.get("test_batch_size", 128), metavar="N")
    parser.add_argument("--epochs", type=int, default=model_defaults.get("epochs", 50), metavar="N")
    parser.add_argument("--model", type=str, default=pre_args.model, choices=list_model_build_config_names())
    parser.add_argument("--lr", type=float, default=model_defaults.get("lr", 6e-5), metavar="LR")
    parser.add_argument("--weight-decay", type=float, default=model_defaults.get("weight_decay", 4e-3), metavar="WD")
    parser.add_argument("--finetune-backbone", action="store_true", default=model_defaults.get("finetune_backbone", True))
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--no-mps", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N")
    parser.add_argument("--data-root", type=str, default=FLOWERS_DATASET_ROOT_DEFAULT)
    parser.add_argument("--no-download", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=model_defaults.get("num_workers", 4), metavar="N")
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default=f"{DEFAULT_MODEL_NAME}_{DATASET_REGISTRY_NAME}")
    parser.add_argument("--wandb-entity", type=str, default="PerforatedAI_IDL")
    parser.add_argument("--wandb-run-name", type=str, default="Normal Gradient Decent")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-api-key", type=str, default="")
    parser.add_argument("--wandb-run-id", type=str, default="")
    parser.add_argument("--wandb-resume", type=str, default="allow", choices=["allow", "must", "never"])
    parser.add_argument("--wandb-anonymous", type=str, default="never", choices=["never", "allow", "must"])
    parser.add_argument("--dendrite-mode", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--max-dendrites", type=int, default=3)
    parser.add_argument("--improvement-threshold", type=float, default=1.0)
    parser.add_argument("--candidate-weight-init-mult", type=float, default=0.1)
    parser.add_argument("--pai-forward-function", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--pai-convert-target", type=str, default=DEFAULT_PAI_CONVERT_TARGET, choices=["pre_fc", "classifier_fc"])
    parser.add_argument("--perforated-load-path", type=str)
    parser.add_argument("--pai-save-name", type=str, default=f"artifacts_{DEFAULT_MODEL_NAME.lower()}_{DATASET_REGISTRY_NAME.lower()}")
    parser.add_argument("--strict-unwrapped-check", action="store_true", default=False)
    parser.add_argument("--strict-weight-decay-check", action="store_true", default=False)
    parser.add_argument("--force-stop-epochs", default=False, action=argparse.BooleanOptionalAction, help="Stops the training after the number mentioned in --epochs flag else it will train until PAI signals to stop")
    parser.add_argument("--gpu", type=int, default=0, metavar="N", help="CUDA device index when CUDA is used (single GPU). Ignored with --no-cuda.")

    args = parser.parse_args()
    model_config = get_model_build_config(args.model)

    default_wandb_project = f"{DEFAULT_MODEL_NAME}_{DATASET_REGISTRY_NAME}"
    if args.wandb_project == default_wandb_project:
        args.wandb_project = f"{args.model}_{DATASET_REGISTRY_NAME}"

    default_pai_save_name = f"artifacts_{DEFAULT_MODEL_NAME.lower()}_{DATASET_REGISTRY_NAME.lower()}"
    if args.pai_save_name == default_pai_save_name:
        args.pai_save_name = f"artifacts_{args.model.lower()}_{DATASET_REGISTRY_NAME.lower()}"

    if GPA is None or UPA is None:
        raise ImportError(
            "PerforatedAI is required for this script. Install it from the PerforatedAI source/package used in your environment."
        )

    # TODO: Do we need this? Wont it be better if this is removed?
    if not args.strict_unwrapped_check and not args.strict_weight_decay_check:
        print("[WARNING] Disabling Interactive Breakpoints (pdb, builtins, sys.exit etc)")
        disable_interactive_breakpoints()

    convert_module_id = selected_convert_module_id(args)
    if args.pai_convert_target != "pre_fc":
        print(
            "Warning: non-n-1 PAI conversion target selected "
            f"({args.pai_convert_target}). This mode is less stable in this workflow."
        )

    use_mps = (not args.no_mps) and torch.backends.mps.is_available()
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    args.gpu = int(args.gpu) if use_cuda else 0
    if use_cuda:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Accelerator currently being used: {device}")

    set_seed(args.seed)

    download = not args.no_download
    print(f"Preparing the following dataset: {DATASET_REGISTRY_NAME}")
    train_transform, val_transform, test_transform, crop_size = model_config.build_transforms()
    dataset_builder = get_dataset_builder(DATASET_REGISTRY_NAME)
    train_dataset, val_dataset, test_dataset = dataset_builder(
        data_root=args.data_root,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        download=download,
    )
    print(f"Dataset Sizes:\n\tTrain: {len(train_dataset)}, \n\tVal: {len(val_dataset)}, \n\tTest: {len(test_dataset)}")

    num_workers = args.num_workers if args.num_workers >= 0 else 0
    print(f"Number of Workers being used in {device}: {num_workers}")
    print(f"Batch Size Information:\n\tTrain batch size: {args.batch_size}, \n\tVal/Test batch size: {args.test_batch_size}")
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
    print("Prepared Dataloaders")
    
    base_model = model_config.build_model(num_classes=NUM_CLASSES, finetune_backbone=args.finetune_backbone)
    model = EfficientNetPAI(base_model)
    print(f"Loaded the following model: {args.model}")

    configure_pai(args, model)
    

    # Note: Rorry suggested to have a single folder name to save instead of passing a path (or abs path)
    pai_system_name = args.pai_save_name
    print(f"Folder name where PAI related things will be stored: {pai_system_name}")

    if not args.perforated_load_path:
        if os.path.exists(pai_system_name):
            print(f"[WARNING] Removing old PAI directory at {pai_system_name}")
            shutil.rmtree(pai_system_name)

    model = UPA.perforate_model(model, save_name=pai_system_name)
    if args.perforated_load_path: 
        # VERY IMPORTANT NOTE: When restarting the script when using DDP - Use switch_x
        model = UPA.load_system(model, pai_system_name, "latest", True)
        # TODO: Load scheduler and optimizer state (if needed) - Make sure to save at appropriate places

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    model = model.to(device)
    optimizer, scheduler = setup_pai_optimizer_scheduler(args, model)

    current_flops = float("nan")
    current_flops_source = "not_computed"
    current_total_params = total_params
    current_trainable_params = trainable_params
    model, current_flops, current_flops_source = refresh_flops_for_training(
        model, device, crop_size
    )

    running_stats: Dict[str, float] = {}
    best_validation_accuracy = float("-inf")
    best_validation_snapshot: Dict[str, float] = {}

    run = init_wandb(args)
    cycle_start = time.perf_counter()

    epoch = 0
    while True:
        epoch += 1
        epoch_start = time.perf_counter()

        train_loss, train_accuracy, train_top5_accuracy = train(args, model, device, train_loader, optimizer, epoch)
        val_metrics, test_metrics = test(model, device, val_loader, test_loader)

         
        GPA.pai_tracker.add_extra_score(train_accuracy, "train")
        GPA.pai_tracker.add_extra_score(train_top5_accuracy, "train_top5")
        # TODO: Can add extra score for val top 5 accuracy

        model, restructured, training_complete = call_pai_add_validation_score(args, float(val_metrics["accuracy"]), model)
        model = model.to(device)
        print(f"Restructured status: {restructured}")
        print(f"Training complete status: {training_complete}")

        if restructured:
            old_optimizer = optimizer
            old_scheduler = scheduler
            optimizer = scheduler = None
            del old_optimizer, old_scheduler
            cuda_gc()
            optimizer, scheduler = setup_pai_optimizer_scheduler(args, model)
            current_total_params, current_trainable_params = count_parameters(model)

            model, current_flops, current_flops_source = refresh_flops_for_training(
                model, device, crop_size
            )

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
            UPA.save_system(model, pai_system_name, "best_model")

        epoch_log: Dict[str, object] = {
            "epoch": epoch,
            "train/accuracy": train_accuracy,
            "train/top5_accuracy": train_top5_accuracy,
            "train/loss": train_loss,
            "val/accuracy": validation_accuracy,
            "val/loss": val_metrics["loss"],
            "val/accuracy_min": running_stats.get("validation_accuracy_min"),
            "val/accuracy_max": running_stats.get("validation_accuracy_max"),
            "val/top5_accuracy": validation_top5,
            "val/top5_min": running_stats.get("validation_top5_min"),
            "val/top5_max": running_stats.get("validation_top5_max"),
            "test/loss": test_metrics["loss"],
            "test/accuracy": test_metrics["accuracy"],
            "test/accuracy_min": running_stats.get("test_accuracy_min"),
            "test/accuracy_max": running_stats.get("test_accuracy_max"),
            "test/top5_accuracy": test_metrics["accuracy_top5"],
            "test/top5_min": running_stats.get("test_top5_min"),
            "test/top5_max": running_stats.get("test_top5_max"),
            "test/precision_at_1": test_metrics["precision_at_1"],
            "seconds_per_training_epoch": seconds_per_training_epoch,
            "seconds_per_training_cycle": seconds_per_training_cycle,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "model/num_parameters": current_total_params,
            "model/trainable_parameters": current_trainable_params,
            "model/flops": safe_number(current_flops),
            "model/flops_source": current_flops_source,
        }

        if hasattr(GPA, "pai_tracker"):
            epoch_log["perforatedai/dendrite_count"] = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)

        if best_validation_snapshot:
            epoch_log["val/test_accuracy_at_best_validation"] = best_validation_snapshot[
                "test_accuracy_at_best_validation"
            ]
            epoch_log["val/test_top5_at_best_validation"] = best_validation_snapshot[
                "test_top5_at_best_validation"
            ]
            epoch_log["val/epoch_at_best_validation"] = best_validation_snapshot["epoch_at_best_validation"]

        print(f"Epoch {epoch} Metrics:\n{json.dumps(epoch_log, indent=3)}")
        if run is not None:
            log_to_wandb(run, epoch_log, step=epoch)

        # Model checkpointing
        UPA.save_system(model, pai_system_name, "latest")

        if training_complete:
            print("PerforatedAI signaled training complete.")
            break
        

        # Pass force_stop_epochs as False if you want to follow PAI signal (training_complete)
        if args.epochs > 0 and args.force_stop_epochs and epoch >= args.epochs:
            print(f"Reached --epochs {args.epochs} safety cap.")
            break

    final_metrics: Dict[str, object] = {}
    model.eval()

    gpu_inference_ips = float("nan")
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_inference_ips = benchmark_inference_throughput(model, gpu_benchmark_loader, device)

    model_cpu = model.to(torch.device("cpu"))
    cpu_inference_ips = benchmark_inference_throughput(model_cpu, cpu_benchmark_loader, torch.device("cpu"))
    latency_ms = benchmark_cpu_latency_single_core_ms(model_cpu, cpu_benchmark_loader)
    param_count, flops, flops_source = compute_model_stats(model_cpu, torch.device("cpu"), crop_size)

    final_metrics = {
        "final/gpu_inference_inputs_per_second": safe_number(gpu_inference_ips),
        "final/cpu_inference_inputs_per_second": safe_number(cpu_inference_ips),
        "final/num_parameters": param_count,
        "final/flops": safe_number(flops),
        "final/flops_source": flops_source,
        "final/latency_ms_per_batch": safe_number(latency_ms),
    }

    if best_validation_snapshot:
        final_metrics["final/accuracy_at_best_validation"] = best_validation_snapshot[
            "test_accuracy_at_best_validation"
        ]
        final_metrics["final/test_top5_at_best_validation"] = best_validation_snapshot[
            "test_top5_at_best_validation"
        ]
        final_metrics["final/validation_accuracy_best"] = best_validation_snapshot[
            "validation_accuracy_best"
        ]
        final_metrics["final/validation_top5_at_best_validation"] = best_validation_snapshot[
            "validation_top5_at_best_validation"
        ]
        final_metrics["final/epoch_at_best_validation"] = best_validation_snapshot["epoch_at_best_validation"]

    print(f"Final performance metrics:\n{json.dumps(final_metrics, indent=3)}")

    UPA.save_system(model, pai_system_name, "final")
    if run is not None:
        if os.path.exists(os.path.join(pai_system_name, "best_model.pt")):
            log_checkpoint_artifact(
                run,
                f"{pai_system_name}-best",
                os.path.join(pai_system_name, "best_model.pt"),
                aliases=["best"],
            )
    pai_png_path = os.path.join(pai_system_name, f"{pai_system_name}.png")
    if os.path.exists(pai_png_path):
        if run is not None and wandb is not None:
            run.log({"perforatedai/pai_graph": wandb.Image(pai_png_path)})
            print(f"PAI graph image logged to W&B from: {pai_png_path}")
    else:
        print(f"PAI graph image not found at: {pai_png_path}")
        

    if run is not None:
        log_to_wandb(run, final_metrics)
        finish_wandb(run)


if __name__ == "__main__":
    main()
