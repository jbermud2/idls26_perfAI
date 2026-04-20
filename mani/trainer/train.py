from __future__ import annotations

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.ddp_utils import fill_missing_parameter_gradients

try:
    from perforatedai import globals_perforatedai as GPA
except ImportError:
    GPA = None


def train(
    args,
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
):
    model.train()
    model.to(device)
    correct = 0
    correct_top5 = 0
    total_loss = 0.0
    total_seen = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        # if args.distributed:
        #     fill_missing_parameter_gradients(model)
        output = model(data)
        loss = F.cross_entropy(output, target)
        print("BEFORE BACKWARD")
        loss.backward()
        print("AFTER BACKWARD")
        if args.perforate_model_parallel:
            if GPA is not None:
                GPA.pai_tracker.save_tracker_settings()
            print(f"[DDP] PAI DDP settings saved to {args.pai_save_name}/")
            print("[DDP] Initialization complete. Now run train_distributed.sh for multi-GPU training.")
            # torch.distributed.barrier()
            sys.exit(0)
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
    correct_value = float(correct.item())
    correct_top5_value = float(correct_top5.item())
    denom = max(int(total_seen), 1)
    train_accuracy = 100.0 * correct_value / denom
    train_top5_accuracy = 100.0 * correct_top5_value / denom
    train_loss = total_loss / denom
    return float(train_loss), float(train_accuracy), float(train_top5_accuracy)
