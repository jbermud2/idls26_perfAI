#python main_baseline.py --data-root /ocean/projects/cis260045p/shared/data --use-wandb --wandb-api-key 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

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.registry import get_dataset_builder
from models import NUM_CLASSES, build_transforms, efficientnet_b5_flowers102
from trainer.eval import test
from trainer.train import train
from utils.generic_utils import set_seed
from utils.metrics_utils import (
    benchmark_cpu_latency_single_core_ms,
    benchmark_inference_throughput,
    compute_model_stats,
    count_parameters,
    refresh_flops_for_training,
    safe_number,
    update_min_max,
)
from utils.wandb_utils import finish_wandb, init_wandb, log_checkpoint_artifact, log_to_wandb


FLOWERS_DATASET_ROOT_DEFAULT = "/ocean/projects/cis260045p/shared/data"
DATASET_REGISTRY_NAME = "flowers102"
MODEL_NAME = "efficientnet_b5"


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch transfer learning with plain EfficientNet-B5 (no PerforatedAI)."
    )
    parser.add_argument("--batch-size", type=int, default=64, metavar="N")
    parser.add_argument("--test-batch-size", type=int, default=128, metavar="N")
    parser.add_argument("--epochs", type=int, default=150, metavar="N")
    parser.add_argument("--lr", type=float, default=1e-4, metavar="LR")
    parser.add_argument("--weight-decay", type=float, default=1e-4, metavar="WD")
    parser.add_argument("--finetune-backbone", action="store_true", default=False)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--no-mps", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N")
    parser.add_argument("--data-root", type=str, default=FLOWERS_DATASET_ROOT_DEFAULT)
    parser.add_argument("--no-download", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=6, metavar="N")
    parser.add_argument("--save-dir", type=str, default="artifacts_efficientnet_b5_flowers102_baseline")

    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default=f"{MODEL_NAME}_{DATASET_REGISTRY_NAME}")
    parser.add_argument("--wandb-entity", type=str, default="PerforatedAI_IDL")
    parser.add_argument("--wandb-run-name", type=str, default="EfficientNet-B5 Baseline")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-api-key", type=str, default="")
    parser.add_argument("--wandb-run-id", type=str, default="")
    parser.add_argument("--wandb-resume", type=str, default="allow", choices=["allow", "must", "never"])
    parser.add_argument("--wandb-anonymous", type=str, default="never", choices=["never", "allow", "must"])

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        metavar="N",
        help="CUDA device index when CUDA is used (single GPU). Ignored with --no-cuda.",
    )

    args = parser.parse_args()

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
    train_transform, val_transform, test_transform, crop_size = build_transforms()
    dataset_builder = get_dataset_builder(DATASET_REGISTRY_NAME)
    train_dataset, val_dataset, test_dataset = dataset_builder(
        data_root=args.data_root,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        download=download,
    )
    print(
        f"Dataset Sizes:\n\tTrain: {len(train_dataset)}, \n\tVal: {len(val_dataset)}, \n\tTest: {len(test_dataset)}"
    )

    num_workers = args.num_workers if args.num_workers >= 0 else 0
    print(f"Number of Workers being used in {device}: {num_workers}")
    print(
        f"Batch Size Information:\n\tTrain batch size: {args.batch_size}, \n\tVal/Test batch size: {args.test_batch_size}"
    )

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

    model = efficientnet_b5_flowers102(
        num_classes=NUM_CLASSES,
        finetune_backbone=args.finetune_backbone,
    )
    print("Loaded plain EfficientNet-B5 baseline model")

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    model = model.to(device)
    current_flops = float("nan")
    current_flops_source = "not_computed"
    model, current_flops, current_flops_source = refresh_flops_for_training(
        model, device, crop_size
    )

    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    if not trainable_param_list:
        raise RuntimeError(
            "No trainable parameters found (all layers are frozen). "
            "Enable trainable layers or run with --finetune-backbone."
        )

    optimizer = optim.AdamW(
        trainable_param_list,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.epochs <= 0:
        raise RuntimeError("--epochs must be > 0 for baseline training.")
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    running_stats: Dict[str, float] = {}
    best_validation_accuracy = float("-inf")
    best_validation_snapshot: Dict[str, float] = {}

    os.makedirs(args.save_dir, exist_ok=True)
    run = init_wandb(args)
    cycle_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy, train_top5_accuracy = train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
        )
        val_metrics, test_metrics = test(model, device, val_loader, test_loader)
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_validation_accuracy": best_validation_accuracy,
                },
                os.path.join(args.save_dir, "best_model.pt"),
            )

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
            "model/num_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/flops": safe_number(current_flops),
            "model/flops_source": current_flops_source,
        }

        if best_validation_snapshot:
            epoch_log["val/test_accuracy_at_best_validation"] = best_validation_snapshot[
                "test_accuracy_at_best_validation"
            ]
            epoch_log["val/test_top5_at_best_validation"] = best_validation_snapshot[
                "test_top5_at_best_validation"
            ]
            epoch_log["val/epoch_at_best_validation"] = best_validation_snapshot[
                "epoch_at_best_validation"
            ]

        print(f"Epoch {epoch} Metrics:\n{json.dumps(epoch_log, indent=3)}")
        if run is not None:
            log_to_wandb(run, epoch_log, step=epoch)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(args.save_dir, "latest.pt"),
        )

        if args.dry_run:
            print("Dry run requested; stopping after first epoch.")
            break

    model.eval()

    gpu_inference_ips = float("nan")
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_inference_ips = benchmark_inference_throughput(model, gpu_benchmark_loader, device)

    model_cpu = model.to(torch.device("cpu"))
    cpu_inference_ips = benchmark_inference_throughput(
        model_cpu,
        cpu_benchmark_loader,
        torch.device("cpu"),
    )
    latency_ms = benchmark_cpu_latency_single_core_ms(model_cpu, cpu_benchmark_loader)
    param_count, flops, flops_source = compute_model_stats(model_cpu, torch.device("cpu"), crop_size)

    final_metrics: Dict[str, object] = {
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
        final_metrics["final/epoch_at_best_validation"] = best_validation_snapshot[
            "epoch_at_best_validation"
        ]

    print(f"Final performance metrics:\n{json.dumps(final_metrics, indent=3)}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model_state_dict.pt"))

    if run is not None and os.path.exists(os.path.join(args.save_dir, "best_model.pt")):
        log_checkpoint_artifact(
            run,
            f"{args.save_dir}-best",
            os.path.join(args.save_dir, "best_model.pt"),
            aliases=["best"],
        )

    if run is not None:
        log_to_wandb(run, final_metrics)
        finish_wandb(run)


if __name__ == "__main__":
    main()
