from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate(
    model: nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
):
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
):
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
