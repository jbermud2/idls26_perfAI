from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist


def setup_ddp(args) -> None:
    args.distributed = False
    args.rank = 0
    args.local_rank = args.gpu
    args.world_size = 1

    if not args.perforate_model_parallel and "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        args.distributed = True
        args.rank = dist.get_rank()
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.world_size = dist.get_world_size()
        print(f"[DDP] Initialized DDP in GPU with local rank {args.local_rank} (rank {args.rank})")


def set_seed_ddp(seed: int = 42) -> None:
    rank = dist.get_rank()
    worker_seed = seed + rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)
    os.environ["PYTHONHASHSEED"] = str(worker_seed)


def build_train_sampler(args, train_dataset):
    if args.distributed:
        return torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
    return None # If not DDP


def wrap_model_ddp(args, model: torch.nn.Module) -> torch.nn.Module:
    if args.distributed:
        return torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )
    return model


def broadcast_pai_flags(restructured: bool, training_complete: bool, device: torch.device) -> tuple[bool, bool]:
    r_tensor = torch.tensor([1 if restructured else 0], dtype=torch.int, device=device)
    tc_tensor = torch.tensor([1 if training_complete else 0], dtype=torch.int, device=device)
    dist.broadcast(r_tensor, src=0)
    dist.broadcast(tc_tensor, src=0)
    return bool(r_tensor.item()), bool(tc_tensor.item())

def sync_training_metrics(val_acc: float, train_acc: float, train_top5: float, device: torch.device) -> tuple[float, float, float]:
    if not dist.is_initialized():
        return val_acc, train_acc, train_top5

    metrics_tensor = torch.tensor([val_acc, train_acc, train_top5], dtype=torch.float32, device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    metrics_tensor /= world_size
 
    return (float(metrics_tensor[0].item()), float(metrics_tensor[1].item()), float(metrics_tensor[2].item()))


def fill_missing_parameter_gradients(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.requires_grad and param.grad is None:
            param.grad = torch.zeros_like(param)


def exit_distributed() -> None:
    # Used when DDP training needs to stop and restart
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    sys.exit(0)


def cleanup_ddp() -> None:
    # Called at the end of a normal (non-exit) run
    if dist.is_initialized():
        dist.destroy_process_group()
