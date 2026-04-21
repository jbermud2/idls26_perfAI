'''
Single GPU:

For running:
interact -p GPU-shared --gres=gpu:v100-32:1 -t 8:00:00 -A cis260045p

interact -p GPU-shared --gres=gpu:h100-80:1 -t 8:00:00 -A cis260045p

Single GPU:
python /ocean/projects/cis260045p/jbermude/idls26_perfAI/mani/main.py --data-root /ocean/projects/cis260045p/shared/data --use-wandb --dendrite-mode 2 --max-dendrites 3 --pai-forward-function relu --improvement-threshold 1 --candidate-weight-init-mult 0.1 > output.txt 2>&1

Perforated Backpropagation (--dendrite-mode 2): install the perforatedbp add-on, then export credentials before Python (do not commit secrets):
  export PAIEMAIL='your@email'
  export PAITOKEN='your_token'
  python .../main.py ... --dendrite-mode 2 ...
(See mani/API/customization.md.)

Multi-GPU DDP:

interact -p GPU-shared --gres=gpu:v100-32:2 -t 8:00:00 -A cis260045p

interact -p GPU-shared --gres=gpu:h100-80:2 -t 8:00:00 -A cis260045p


./train_distributed.sh --data-root /ocean/projects/cis260045p/shared/data --use-wandb --wandb-api-key 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 --dendrite-mode 1 --max-dendrites 3 --pai-forward-function relu --improvement-threshold 1 --candidate-weight-init-mult 0.1 --epochs 25 > output.txt 2>&1

./train_distributed.sh --data-root /ocean/projects/cis260045p/shared/data --dendrite-mode 1 --max-dendrites 3 --pai-forward-function relu --improvement-threshold 1 --candidate-weight-init-mult 0.1 --epochs 25 2>&1 | tee output.txt
'''

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import shutil
from typing import Dict
import glob

import torch
import torch.distributed as dist

from data.registry import get_dataset_builder
# from models import EfficientNetB5PAI, NUM_CLASSES_PETS, build_transforms, efficientnet_b5_pets
from models.efficientnet import (
    EfficientNetB5PAI,
    NUM_CLASSES_PETS,
    build_transforms,
    efficientnet_b5_pets,
)
from trainer.eval import test
from trainer.train import train
from utils.ddp_utils import (
    broadcast_pai_flags,
    sync_training_metrics,
    build_train_sampler,
    cleanup_ddp,
    exit_distributed,
    set_seed_ddp,
    setup_ddp,
    wrap_model_ddp,
)
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
    safe_number,
    update_min_max,
)
from utils.pai_utils import (
    configure_pai,
    selected_convert_module_id,
    setup_pai_optimizer_scheduler
)
from utils.wandb_utils import (
    finish_wandb,
    init_wandb,
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


PETS_DATASET_ROOT_DEFAULT = "/ocean/projects/cis260045p/shared/data"
DEFAULT_PAI_CONVERT_TARGET = "pre_fc"
DATASET_REGISTRY_NAME = "pets"
MODEL_NAME = "efficientnet_b5"


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch transfer learning with EfficientNet-B5 + PerforatedAI (DDP support)."
    )
    parser.add_argument("--batch-size", type=int, default=64, metavar="N") # changed to 64 using Olivia's advice
    parser.add_argument("--test-batch-size", type=int, default=128, metavar="N")
    parser.add_argument("--epochs", type=int, default=100, metavar="N")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--weight-decay", type=float, default=1e-4, metavar="WD")
    parser.add_argument("--finetune-backbone", action="store_true", default=False)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--no-mps", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N")
    parser.add_argument("--data-root", type=str, default=PETS_DATASET_ROOT_DEFAULT)
    parser.add_argument("--no-download", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=6, metavar="N")
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default=f"{MODEL_NAME}_{DATASET_REGISTRY_NAME}")
    parser.add_argument("--wandb-entity", type=str, default="PerforatedAI_IDL")
    parser.add_argument("--wandb-run-name", type=str, default="EfficientNet_B5_Pets")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-api-key", type=str, default="")
    parser.add_argument("--wandb-run-id", type=str, default="")
    parser.add_argument("--wandb-resume", type=str, default="allow", choices=["allow", "must", "never"])
    parser.add_argument("--wandb-anonymous", type=str, default="never", choices=["never", "allow", "must"])
    parser.add_argument(
        "--dendrite-mode",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="0=no dendrites, 1=PAI without Perforated BP, 2=Perforated Backpropagation (requires perforatedbp + PAITOKEN/PAIEMAIL env).",
    )
    parser.add_argument("--max-dendrites", type=int, default=3)
    parser.add_argument("--improvement-threshold", type=float, default=1.0)
    parser.add_argument("--candidate-weight-init-mult", type=float, default=0.1)
    parser.add_argument("--pai-forward-function", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--pai-convert-target", type=str, default=DEFAULT_PAI_CONVERT_TARGET, choices=["pre_fc", "classifier_fc"])
    parser.add_argument("--perforated-load-path", type=str)
    parser.add_argument("--pai-save-name", type=str, default=f"artifacts_{MODEL_NAME.lower()}_{DATASET_REGISTRY_NAME.lower()}")
    parser.add_argument("--strict-unwrapped-check", action="store_true", default=False)
    parser.add_argument("--strict-weight-decay-check", action="store_true", default=False)
    parser.add_argument("--force-stop-epochs", default=False, action=argparse.BooleanOptionalAction, help="Stops the training after the number mentioned in --epochs flag else it will train until PAI signals to stop")
    parser.add_argument("--gpu", type=int, default=0, metavar="N", help="CUDA device index when CUDA is used (single GPU). Ignored with --no-cuda.")
    parser.add_argument("--perforate_model_parallel", action="store_true",
                        help="Initialize PAI settings for DDP (run once on single GPU, handled by train_distributed.sh)")
    parser.add_argument("--pai_load_folder", type=str, default=None,
                        help="Folder to load PAI state from for DDP automatic resumption after dendrite restructure")

    args = parser.parse_args()

    # Initialize DDP (sets args.distributed, args.rank, args.local_rank, args.world_size)
    setup_ddp(args)

    if GPA is None or UPA is None:
        raise ImportError(
            "PerforatedAI is required for this script. Install it from the PerforatedAI source/package used in your environment."
        )

    # if args.dendrite_mode == 2:
    #     try:
    #         import perforatedbp  # noqa: F401
    #     except ImportError as exc:
    #         raise ImportError(
    #             "Perforated Backpropagation (--dendrite-mode 2) requires the `perforatedbp` "
    #             "package. Install the Perforated Backpropagation add-on in this environment."
    #         ) from exc
    #     if not os.environ.get("PAITOKEN") or not os.environ.get("PAIEMAIL"):
    #         raise RuntimeError(
    #             "Perforated Backpropagation (--dendrite-mode 2) requires PAITOKEN and "
    #             "PAIEMAIL to be set in the environment before launching Python "
    #             "(see mani/API/customization.md)."
    #         )

    # Only disable interactive breakpoints in single-GPU non-init mode;
    # DDP and init mode need sys.exit() to work for controlled restarts.
    if not args.strict_unwrapped_check and not args.strict_weight_decay_check:
        if not args.distributed and not args.perforate_model_parallel:
            print("[WARNING] Disabling Interactive Breakpoints (pdb, builtins, sys.exit etc)")
            disable_interactive_breakpoints()

    convert_module_id = selected_convert_module_id(args)
    if args.pai_convert_target != "pre_fc" and args.rank == 0:
        print(
            "Warning: non-n-1 PAI conversion target selected "
            f"({args.pai_convert_target}). This mode is less stable in this workflow."
        )

    use_mps = (not args.no_mps) and torch.backends.mps.is_available()
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if args.rank == 0:
        print(f"Accelerator currently being used: {device}")

    if args.distributed:
        set_seed_ddp(args.seed)
    else:
        set_seed(args.seed)

    download = not args.no_download
    if args.rank == 0:
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
    if args.rank == 0:
        print(f"Dataset Sizes:\n\tTrain: {len(train_dataset)}, \n\tVal: {len(val_dataset)}, \n\tTest: {len(test_dataset)}")

    num_workers = args.num_workers if args.num_workers >= 0 else 0
    if args.rank == 0:
        print(f"Number of Workers being used in {device}: {num_workers}")
        print(f"Batch Size Information:\n\tTrain batch size: {args.batch_size}, \n\tVal/Test batch size: {args.test_batch_size}")

    train_sampler = build_train_sampler(args, train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0 and args.distributed),
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
    if args.rank == 0:
        print("Prepared Dataloaders")

    base_model = efficientnet_b5_pets(num_classes=NUM_CLASSES_PETS, finetune_backbone=args.finetune_backbone)
    model = EfficientNetB5PAI(base_model)
    if args.rank == 0:
        print(f"Loaded the following model: {MODEL_NAME}")

    configure_pai(args, model)

    pai_system_name = args.pai_save_name
    if args.rank == 0:
        print(f"[DDP] Folder name where PAI related things will be stored: {pai_system_name}")

    if not args.perforated_load_path and args.pai_load_folder is None and not args.distributed:
        if os.path.exists(pai_system_name) and args.rank == 0:
            print(f"[WARNING] Removing old PAI directory at {pai_system_name}")
            shutil.rmtree(pai_system_name)

    model = UPA.perforate_model(model, save_name=pai_system_name)

    # DDP restart: load the highest-numbered switch checkpoint saved before a dendrite restructure
    if args.pai_load_folder is not None:
        switch_files = glob.glob(f"{args.pai_load_folder}/switch_*.pt")
        if switch_files:
            switch_numbers = []
            for f in switch_files:
                try:
                    num = int(f.split("switch_")[1].split(".pt")[0])
                    switch_numbers.append(num)
                except Exception:
                    pass
            if switch_numbers:
                max_switch = max(switch_numbers)
                model = UPA.load_system(model, args.pai_load_folder, f"switch_{max_switch}", True)
                if args.rank == 0:
                    print(f"[DDP] Loaded PAI state from {args.pai_load_folder}/switch_{max_switch}.pt")
            elif args.rank == 0:
                print(f"[DDP] Starting from beginning (no valid switch_x.pt found in {args.pai_load_folder})")
        elif args.rank == 0:
            print(f"[DDP] Starting from beginning (no switch_x.pt files found in {args.pai_load_folder})")
    elif args.perforated_load_path:
        model = UPA.load_system(model, pai_system_name, "latest", True)

    if args.rank == 0:
        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    model = model.to(device)
    optimizer, scheduler = setup_pai_optimizer_scheduler(args, model)

    if not args.perforate_model_parallel:
        if args.distributed:
            GPA.pai_tracker.initialize_tracker_settings()
        model = wrap_model_ddp(args, model)

    running_stats: Dict[str, float] = {}
    best_validation_accuracy = float("-inf")
    best_validation_snapshot: Dict[str, float] = {}

    run = init_wandb(args) if args.rank == 0 else None
    cycle_start = time.perf_counter()

    epoch = 0
    while True:
        epoch += 1
        epoch_start = time.perf_counter()
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy, train_top5_accuracy = train(args, model, device, train_loader, optimizer, epoch)
        print("I FINISHED TRAINING")
        val_metrics, test_metrics = test(model, device, val_loader, test_loader)



        if args.distributed:
            synced_val_acc, synced_train_acc, synced_train_top5 = sync_training_metrics(float(val_metrics["accuracy"]), train_accuracy, train_top5_accuracy, device)
            model_unwrapped = model.module
            # GPA.pai_tracker.add_extra_score(synced_train_acc, "train")
            # GPA.pai_tracker.add_extra_score(synced_train_top5, "train_top5")
            # _, restructured, training_complete = GPA.pai_tracker.add_validation_score(synced_val_acc, model_unwrapped)
            # if args.rank != 0:
            #     restructured = False
            #     training_complete = False

            # torch.distributed.barrier()
            # restructured, training_complete = broadcast_pai_flags(restructured, training_complete, device)

            # 2. Rank 0 runs the tracker FIRST to see if file I/O is needed
            if args.rank == 0:
                GPA.pai_tracker.add_extra_score(synced_train_acc, "train")
                GPA.pai_tracker.add_extra_score(synced_train_top5, "train_top5")
                _, restructured, training_complete = GPA.pai_tracker.add_validation_score(synced_val_acc, model_unwrapped)
            else:
                restructured = False
                training_complete = False

            restructured, training_complete = broadcast_pai_flags(restructured, training_complete, device)

            if args.rank != 0 and not restructured and not training_complete:
                GPA.pai_tracker.add_extra_score(synced_train_acc, "train")
                GPA.pai_tracker.add_extra_score(synced_train_top5, "train_top5")
                _, _, _ = GPA.pai_tracker.add_validation_score(synced_val_acc, model_unwrapped)
        else:
            GPA.pai_tracker.add_extra_score(train_accuracy, "train")
            GPA.pai_tracker.add_extra_score(train_top5_accuracy, "train_top5")
            model, restructured, training_complete = GPA.pai_tracker.add_validation_score(float(val_metrics["accuracy"]), model)
            model = model.to(device)

        if args.rank == 0:
            print(f"Restructured status: {restructured}")
            print(f"Training complete status: {training_complete}")
        if training_complete:
            if args.distributed:
                if args.rank == 0:
                    print("PAI training complete!")
                    os.makedirs(pai_system_name, exist_ok=True)
                    with open(f"{pai_system_name}/.training_complete", "w") as f:
                        f.write("complete")
                exit_distributed()
            # Non-DDP: fall through to the training_complete break below

        elif restructured:
            if args.distributed:
                if args.rank == 0:
                    print("Model restructured! Exiting for DDP restart...")
                exit_distributed()
            else:
                old_optimizer = optimizer
                old_scheduler = scheduler
                optimizer = scheduler = None
                del old_optimizer, old_scheduler
                cuda_gc()
                optimizer, scheduler = setup_pai_optimizer_scheduler(args, model)

        seconds_per_training_epoch = time.perf_counter() - epoch_start
        seconds_per_training_cycle = time.perf_counter() - cycle_start

        validation_accuracy = val_metrics["accuracy"]
        validation_top5 = val_metrics["accuracy_top5"]

        update_min_max(running_stats, "validation_accuracy", validation_accuracy)
        update_min_max(running_stats, "validation_top5", validation_top5)
        update_min_max(running_stats, "test_accuracy", test_metrics["accuracy"])
        update_min_max(running_stats, "test_top5", test_metrics["accuracy_top5"])

        if args.rank == 0:
            model_for_count = model.module if args.distributed else model
            total_params, trainable_params = count_parameters(model_for_count)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

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
            "learning_rate": current_lr,
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

        if args.rank == 0:
            print(f"Epoch {epoch} Metrics:\n{json.dumps(epoch_log, indent=3)}")
        if run is not None:
            log_to_wandb(run, epoch_log, step=epoch)

        # Checkpoint from rank 0 only (all ranks have identical weights)
        if args.rank == 0:
            model_to_save = model.module if args.distributed else model
            UPA.save_system(model_to_save, pai_system_name, "latest")

        if training_complete:
            if args.rank == 0:
                print("PerforatedAI signaled training complete.")
            break

        if args.epochs > 0 and args.force_stop_epochs and epoch >= args.epochs:
            if args.rank == 0:
                print(f"Reached --epochs {args.epochs} safety cap.")
            break

    cleanup_ddp()

    if args.rank == 0:
        final_metrics: Dict[str, object] = {}
        model_for_metrics = model.module if args.distributed else model
        model_for_metrics.eval()

        gpu_inference_ips = float("nan")
        if torch.cuda.is_available() and not args.no_cuda:
            gpu_inference_ips = benchmark_inference_throughput(model_for_metrics, gpu_benchmark_loader, device)

        model_cpu = model_for_metrics.to(torch.device("cpu"))
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

        model_to_save = model.module if args.distributed else model
        UPA.save_system(model_to_save, pai_system_name, "final")
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
