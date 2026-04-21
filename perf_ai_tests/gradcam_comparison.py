"""
GradCAM Comparison Script: Baseline vs Perforated EfficientNet-B5
=================================================================
Works with:
  - main_baseline.py  → saves plain efficientnet_b5_flowers102 to save_dir/best_model.pt
  - main.py        → saves EfficientNetB5PAI to output_dir/efficientnet_b5_*_best.pt

Usage on PSC Bridges2:
----------------------
pip install grad-cam --quiet

# Pets
python gradcam_comparison.py \
    --data-root /ocean/projects/cis260045p/shared/data \
    --output-dir ./gradcam_outputs \
    --datasets pets \
    --pets-baseline-ckpt   /ocean/projects/cis260045p/labebe/outputs_pets_baseline_v2/best_model.pt \
    --pets-perforated-ckpt /ocean/projects/cis260045p/labebe/outputs_pets_perforated_v3/efficientnet_b5_pets_best.pt

# Flowers102
python gradcam_comparison.py \
    --data-root /ocean/projects/cis260045p/shared/data \
    --output-dir ./gradcam_outputs \
    --datasets flowers102 \
    --flowers102-baseline-ckpt   /ocean/projects/cis260045p/labebe/outputs_flowers_baseline/best_model.pt \
    --flowers102-perforated-ckpt /ocean/projects/cis260045p/labebe/outputs_flowers_efficientnet/efficientnet_b5_flowers102_best.pt

# All datasets at once
python gradcam_comparison.py \
    --data-root /ocean/projects/cis260045p/shared/data \
    --output-dir ./gradcam_outputs \
    --datasets flowers102 pets food101 \
    --flowers102-baseline-ckpt   /ocean/projects/cis260045p/labebe/outputs_flowers_baseline/best_model.pt \
    --flowers102-perforated-ckpt /ocean/projects/cis260045p/labebe/outputs_flowers_efficientnet/efficientnet_b5_flowers102_best.pt \
    --pets-baseline-ckpt         /ocean/projects/cis260045p/labebe/outputs_pets_baseline_v2/best_model.pt \
    --pets-perforated-ckpt       /ocean/projects/cis260045p/labebe/outputs_pets_perforated_v3/efficientnet_b5_pets_best.pt \
    --food101-baseline-ckpt      /ocean/projects/cis260045p/labebe/outputs_food_baseline/best_model.pt \
    --food101-perforated-ckpt    /ocean/projects/cis260045p/labebe/outputs_food_perforated/efficientnet_b5_food101_best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    print("ERROR: pytorch-grad-cam is not installed. Run: pip install grad-cam")
    sys.exit(1)

try:
    from models import EfficientNetB5PAI, build_transforms, efficientnet_b5_flowers102
    from data.registry import get_dataset_builder
except ImportError as e:
    print(f"ERROR importing project modules: {e}")
    print("Make sure you are cd'd into ~/idls26_perfAI/ and conda env is activated.")
    sys.exit(1)


# ------------------------------------------------------------------ #
#  Dataset config
# ------------------------------------------------------------------ #
DATASET_CONFIG = {
    "flowers102": {"num_classes": 102},
    "pets":       {"num_classes": 37},
    "food101":    {"num_classes": 101},
}

NUM_CORRECT   = 5
NUM_INCORRECT = 3


# ------------------------------------------------------------------ #
#  Model loaders — one for baseline, one for perforated
# ------------------------------------------------------------------ #
def load_baseline_model(checkpoint_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """
    Load plain efficientnet_b5_flowers102 saved by main_baseline.py.
    Checkpoint format: {"model_state_dict": ..., "epoch": ..., ...}
    or raw state dict.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")

    # Plain model — no PAI wrapper
    model = efficientnet_b5_flowers102(num_classes=num_classes, finetune_backbone=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Filter to only keys that exist in the model
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    missing = model_keys - set(filtered.keys())
    if missing:
        print(f"  WARNING: {len(missing)} keys missing, loading with strict=False")
        model.load_state_dict(filtered, strict=False)
    else:
        model.load_state_dict(filtered, strict=True)

    model.eval()
    model.to(device)
    print(f"  Loaded baseline: {checkpoint_path} ({num_classes} classes)")
    return model


def load_perforated_model(checkpoint_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """
    Load EfficientNetB5PAI saved by main(4).py.
    Checkpoint is a raw state dict (saved via state_dict_without_pai_metadata).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Perforated checkpoint not found: {checkpoint_path}")

    base = efficientnet_b5_flowers102(num_classes=num_classes, finetune_backbone=False)
    model = EfficientNetB5PAI(base)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    missing = model_keys - set(filtered.keys())
    if missing:
        print(f"  WARNING: {len(missing)} keys missing, loading with strict=False")
        model.load_state_dict(filtered, strict=False)
    else:
        model.load_state_dict(filtered, strict=True)

    model.eval()
    model.to(device)
    print(f"  Loaded perforated: {checkpoint_path} ({num_classes} classes)")
    return model


# ------------------------------------------------------------------ #
#  GradCAM target layer
#  - Baseline: plain EfficientNet → model.features[-1]
#  - Perforated: EfficientNetB5PAI → model.features[-1] (same attr name)
# ------------------------------------------------------------------ #
def get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "features"):
        layer = model.features[-1]
        print(f"  GradCAM target: model.features[-1] = {type(layer).__name__}")
        return layer
    raise AttributeError("Cannot find 'features' on model.")


# ------------------------------------------------------------------ #
#  Collect correct + incorrect examples
# ------------------------------------------------------------------ #
@torch.no_grad()
def collect_examples(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    n_correct: int,
    n_incorrect: int,
) -> Tuple[List[int], List[int]]:
    correct, incorrect = [], []
    for idx in range(len(dataset)):
        if len(correct) >= n_correct and len(incorrect) >= n_incorrect:
            break
        img, label = dataset[idx]
        if not isinstance(img, torch.Tensor):
            raise RuntimeError("Expected tensor from dataset — check val_transform.")
        inp = img.unsqueeze(0).to(device)
        pred = model(inp).argmax(dim=1).item()
        if pred == label and len(correct) < n_correct:
            correct.append(idx)
        elif pred != label and len(incorrect) < n_incorrect:
            incorrect.append(idx)
    return correct, incorrect


# ------------------------------------------------------------------ #
#  Run GradCAM on one image
# ------------------------------------------------------------------ #
def run_gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    img_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> Tuple[np.ndarray, int, float]:
    cam = GradCAM(model=model, target_layers=[target_layer])
    inp = img_tensor.to(device)
    grayscale = cam(input_tensor=inp, targets=[ClassifierOutputTarget(target_class)])[0]

    rgb = inp.squeeze(0).cpu().permute(1, 2, 0).numpy()
    rgb = np.clip(np.array([0.229, 0.224, 0.225]) * rgb + np.array([0.485, 0.456, 0.406]), 0, 1).astype(np.float32)
    overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

    with torch.no_grad():
        probs = F.softmax(model(inp), dim=1)
        pred  = probs.argmax(dim=1).item()
        conf  = probs[0, pred].item()

    return overlay, pred, conf


# ------------------------------------------------------------------ #
#  Build and save comparison figure
# ------------------------------------------------------------------ #
def make_figure(
    baseline_model,
    perforated_model,
    dataset,
    device,
    dataset_name,
    output_dir,
    n_correct=NUM_CORRECT,
    n_incorrect=NUM_INCORRECT,
):
    print(f"\n[{dataset_name}] Collecting examples...")
    correct_idx, incorrect_idx = collect_examples(
        baseline_model, dataset, device, n_correct, n_incorrect
    )
    all_idx   = correct_idx + incorrect_idx
    all_types = ["correct"] * len(correct_idx) + ["incorrect"] * len(incorrect_idx)

    if not all_idx:
        print(f"  WARNING: No examples found for {dataset_name}, skipping.")
        return

    baseline_layer   = get_target_layer(baseline_model)
    perforated_layer = get_target_layer(perforated_model)

    n = len(all_idx)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n), squeeze=False)
    fig.suptitle(
        f"GradCAM: Baseline vs Perforated EfficientNet-B5\nDataset: {dataset_name}",
        fontsize=14, fontweight="bold",
    )
    for col, title in enumerate(["Original Image", "Baseline (no dendrites)", "Perforated (with dendrites)"]):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for row, (idx, ltype) in enumerate(zip(all_idx, all_types)):
        img_tensor, true_label = dataset[idx]
        inp = img_tensor.unsqueeze(0)

        # Original (denormalized)
        rgb = img_tensor.permute(1, 2, 0).numpy()
        rgb = np.clip(np.array([0.229, 0.224, 0.225]) * rgb + np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[row][0].imshow(rgb)
        axes[row][0].set_ylabel(
            f"{'✓ Correct' if ltype == 'correct' else '✗ Wrong'}\ntrue={true_label}",
            fontsize=9, rotation=0, labelpad=65, va="center",
        )
        axes[row][0].axis("off")

        # Baseline GradCAM
        cam_b, pred_b, conf_b = run_gradcam(baseline_model, baseline_layer, inp, true_label, device)
        axes[row][1].imshow(cam_b)
        axes[row][1].set_title(
            f"pred={pred_b}  conf={conf_b:.1%}",
            fontsize=8, color="green" if pred_b == true_label else "red",
        )
        axes[row][1].axis("off")

        # Perforated GradCAM
        cam_p, pred_p, conf_p = run_gradcam(perforated_model, perforated_layer, inp, true_label, device)
        axes[row][2].imshow(cam_p)
        axes[row][2].set_title(
            f"pred={pred_p}  conf={conf_p:.1%}",
            fontsize=8, color="green" if pred_p == true_label else "red",
        )
        axes[row][2].axis("off")

        print(f"  [{ltype}] idx={idx} true={true_label} | "
              f"base={pred_b}({conf_b:.1%}) | perf={pred_p}({conf_p:.1%})")

    plt.tight_layout()
    out = os.path.join(output_dir, f"gradcam_{dataset_name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="GradCAM: baseline vs perforated EfficientNet-B5")
    parser.add_argument("--data-root",   type=str, default="./data")
    parser.add_argument("--output-dir",  type=str, default="./gradcam_outputs")
    parser.add_argument("--datasets",    nargs="+", default=["pets"],
                        choices=["flowers102", "pets", "food101"])
    parser.add_argument("--no-cuda",     action="store_true", default=False)
    parser.add_argument("--n-correct",   type=int, default=NUM_CORRECT)
    parser.add_argument("--n-incorrect", type=int, default=NUM_INCORRECT)

    # Per-dataset checkpoint args
    parser.add_argument("--flowers102-baseline-ckpt",   type=str, default=None)
    parser.add_argument("--flowers102-perforated-ckpt", type=str, default=None)
    parser.add_argument("--pets-baseline-ckpt",         type=str, default=None)
    parser.add_argument("--pets-perforated-ckpt",       type=str, default=None)
    parser.add_argument("--food101-baseline-ckpt",      type=str, default=None)
    parser.add_argument("--food101-perforated-ckpt",    type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_map = {
        "flowers102": (args.flowers102_baseline_ckpt,   args.flowers102_perforated_ckpt),
        "pets":       (args.pets_baseline_ckpt,         args.pets_perforated_ckpt),
        "food101":    (args.food101_baseline_ckpt,       args.food101_perforated_ckpt),
    }

    _, val_transform, _, _ = build_transforms()

    for ds in args.datasets:
        print(f"\n{'='*60}\nDataset: {ds}\n{'='*60}")

        baseline_ckpt, perforated_ckpt = ckpt_map[ds]
        num_classes = DATASET_CONFIG[ds]["num_classes"]

        if not baseline_ckpt:
            print(f"  SKIP: --{ds}-baseline-ckpt not provided.")
            continue
        if not perforated_ckpt:
            print(f"  SKIP: --{ds}-perforated-ckpt not provided.")
            continue
        if not os.path.exists(baseline_ckpt):
            print(f"  SKIP: baseline checkpoint not found: {baseline_ckpt}")
            continue
        if not os.path.exists(perforated_ckpt):
            print(f"  SKIP: perforated checkpoint not found: {perforated_ckpt}")
            continue

        print("Loading baseline model (plain EfficientNet)...")
        baseline_model = load_baseline_model(baseline_ckpt, num_classes, device)

        print("Loading perforated model (EfficientNetB5PAI)...")
        perforated_model = load_perforated_model(perforated_ckpt, num_classes, device)

        dataset_builder = get_dataset_builder(ds)
        _, val_dataset, _ = dataset_builder(
            data_root=args.data_root,
            train_transform=val_transform,
            val_transform=val_transform,
            test_transform=val_transform,
            download=False,
        )
        print(f"  Val set: {len(val_dataset)} images")

        make_figure(
            baseline_model=baseline_model,
            perforated_model=perforated_model,
            dataset=val_dataset,
            device=device,
            dataset_name=ds,
            output_dir=args.output_dir,
            n_correct=args.n_correct,
            n_incorrect=args.n_incorrect,
        )

    print(f"\nDone. Figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
