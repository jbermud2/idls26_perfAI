"""
GradCAM visualization for all EfficientNet-B4 Flowers102 runs.
Uses the WandB API to download checkpoints automatically — no manual downloads needed.

Usage:
------
# Run GradCAM for ALL 12 runs (produces one .png per run)
python gradcam_b4.py \
    --data-root /ocean/projects/cis260045p/shared/data \
    --output-dir /ocean/projects/cis260045p/labebe/gradcam_outputs

# Run for a single specific run by name
python gradcam_b4.py \
    --data-root /ocean/projects/cis260045p/shared/data \
    --output-dir /ocean/projects/cis260045p/labebe/gradcam_outputs \
    --run-name "NGD B4 Baseline"

# Checkpoints are cached in --cache-dir so re-runs skip downloading
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── GradCAM ───────────────────────────────────────────────────────────────────
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    print("ERROR: pip install grad-cam")
    sys.exit(1)

# ── SafeTensors ───────────────────────────────────────────────────────────────
try:
    from safetensors.torch import load_file as safetensors_load
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# ── WandB ─────────────────────────────────────────────────────────────────────
try:
    import wandb
except ImportError:
    print("ERROR: pip install wandb")
    sys.exit(1)

# ── Project imports ───────────────────────────────────────────────────────────
try:
    from models.efficientnet_b4 import efficientnet_b4_flowers102, build_transforms_efficientnet_b4
    from models.efficientnet_common import EfficientNetPAI, NUM_CLASSES
    from data.registry import get_dataset_builder
except ImportError as e:
    print(f"ERROR: {e}")
    print("Run from ~/idls26_perfAI/perf_ai_tests/ with the perf_ai conda env activated.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# All 12 runs: (display_name, artifact_name, model_type)
# model_type "baseline"   = standard .pt with model_state_dict key
# model_type "perforated" = SafeTensors with PAI structure
# ─────────────────────────────────────────────────────────────────────────────
ENTITY  = "PerforatedAI_IDL"
PROJECT = "efficientnet_b4_flowers102"

ALL_RUNS = [
    ("NGD B4 Baseline",             "artifacts_efficientnet_b4_flowers102_baseline-best:v0", "baseline"),
    ("PerforatedBP B4 - 2.3 (0.1)", "artifacts_efficientnet_b4_flowers102-best:v4",          "perforated"),
    ("NGD B4 - 3 Dendrites",        "artifacts_efficientnet_b4_flowers102-best:v5",           "perforated"),
    ("NGD B4 - 2 Dendrites",        "artifacts_efficientnet_b4_flowers102-best:v6",           "perforated"),
    ("NGD B4 - 1 Dendrite",         "artifacts_efficientnet_b4_flowers102-best:v7",           "perforated"),
    ("100 epoch NGD B4 - 3",        "artifacts_efficientnet_b4_flowers102-best:v8",           "perforated"),
    ("PerforatedBP B4 - 3 (0.2)",   "artifacts_efficientnet_b4_flowers102-best:v9",           "perforated"),
    ("PerforatedBP - 3 (0.05)",     "artifacts_efficientnet_b4_flowers102-best:v10",          "perforated"),
    ("PerforatedBP - 1",            "artifacts_efficientnet_b4_flowers102-best:v11",          "perforated"),
    ("PerforatedBP - 2",            "artifacts_efficientnet_b4_flowers102-best:v12",          "perforated"),
    ("PerforatedBP - 1.2",          "artifacts_efficientnet_b4_flowers102-best:v14",          "perforated"),
    ("PerforatedBP 3.4",            "artifacts_efficientnet_b4_flowers102-best:v15",          "perforated"),
]


# ─────────────────────────────────────────────────────────────────────────────
# WandB artifact download
# ─────────────────────────────────────────────────────────────────────────────

def download_artifact(api, artifact_name, cache_dir):
    """Download artifact and return path to best_model.pt. Uses cache to skip re-downloads."""
    safe_name = artifact_name.replace(":", "_").replace("/", "_")
    local_dir = os.path.join(cache_dir, safe_name)
    pt_path   = os.path.join(local_dir, "best_model.pt")

    if os.path.exists(pt_path):
        print(f"  [cached] {pt_path}")
        return pt_path

    print(f"  Downloading: {ENTITY}/{PROJECT}/{artifact_name} ...")
    artifact = api.artifact(f"{ENTITY}/{PROJECT}/{artifact_name}")
    artifact.download(root=local_dir)
    print(f"  Saved to: {local_dir}")
    return pt_path


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_baseline_model(checkpoint_path, device):
    print(f"  [baseline] {checkpoint_path}")
    model = efficientnet_b4_flowers102(num_classes=NUM_CLASSES, finetune_backbone=True)
    ck    = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd    = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(sd, strict=True)
    print(f"  All keys matched.")
    return model.eval().to(device)


def load_perforated_model(checkpoint_path, device):
    print(f"  [perforated] {checkpoint_path}")
    if not HAS_SAFETENSORS:
        print("ERROR: pip install safetensors")
        sys.exit(1)
    base  = efficientnet_b4_flowers102(num_classes=NUM_CLASSES, finetune_backbone=True)
    model = EfficientNetPAI(base)
    sd    = safetensors_load(checkpoint_path, device=str(device))
    model_keys = set(model.state_dict().keys())
    filtered   = {k: v for k, v in sd.items() if k in model_keys}
    missing    = model_keys - set(filtered.keys())
    print(f"  Matched {len(filtered)} keys | {len(missing)} missing "
          f"| {len(sd) - len(filtered)} PAI metadata tensors ignored.")
    model.load_state_dict(filtered, strict=False)
    return model.eval().to(device)


def load_model(checkpoint_path, model_type, device):
    if model_type == "baseline":
        return load_baseline_model(checkpoint_path, device)
    return load_perforated_model(checkpoint_path, device)


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_target_layer(model):
    if hasattr(model, "features"):
        return model.features[-1]
    raise AttributeError("Cannot find 'features' on model.")


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().permute(1, 2, 0).numpy()
    return np.clip(std * img + mean, 0, 1).astype(np.float32)


def run_gradcam(model, target_layer, img_tensor, target_class, device):
    cam = GradCAM(model=model, target_layers=[target_layer])
    inp = img_tensor.to(device)
    inp.requires_grad_(True)
    with torch.enable_grad():
        grayscale = cam(input_tensor=inp,
                        targets=[ClassifierOutputTarget(target_class)])[0]
    rgb     = denormalize(inp.squeeze(0).detach())
    overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
    with torch.no_grad():
        probs = F.softmax(model(inp), dim=1)
        pred  = probs.argmax(dim=1).item()
        conf  = probs[0, pred].item()
    return overlay, rgb, pred, conf


# ─────────────────────────────────────────────────────────────────────────────
# Example collection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_examples(model, dataset, device, n_correct=4, n_incorrect=3):
    correct, incorrect = [], []
    for idx in range(len(dataset)):
        if len(correct) >= n_correct and len(incorrect) >= n_incorrect:
            break
        img, label = dataset[idx]
        pred = model(img.unsqueeze(0).to(device)).argmax(dim=1).item()
        if pred == label and len(correct) < n_correct:
            correct.append(idx)
        elif pred != label and len(incorrect) < n_incorrect:
            incorrect.append(idx)
    print(f"  Collected {len(correct)} correct + {len(incorrect)} incorrect examples.")
    return correct, incorrect


# ─────────────────────────────────────────────────────────────────────────────
# Figure builder
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(model, dataset, device, output_path, run_name,
                n_correct=4, n_incorrect=3):
    target_layer = get_target_layer(model)
    correct_idx, incorrect_idx = collect_examples(
        model, dataset, device, n_correct, n_incorrect)
    all_idx   = correct_idx + incorrect_idx
    all_types = ["correct"] * len(correct_idx) + ["incorrect"] * len(incorrect_idx)

    n = len(all_idx)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n), squeeze=False)
    fig.suptitle(f"GradCAM — {run_name}\nEfficientNet-B4  Flowers102",
                 fontsize=12, fontweight="bold")
    axes[0][0].set_title("Original Image",  fontsize=10, fontweight="bold")
    axes[0][1].set_title("GradCAM Heatmap", fontsize=10, fontweight="bold")

    for row, (idx, ltype) in enumerate(zip(all_idx, all_types)):
        img_tensor, true_label = dataset[idx]
        overlay, rgb, pred, conf = run_gradcam(
            model, target_layer, img_tensor.unsqueeze(0), true_label, device)
        axes[row][0].imshow(rgb)
        axes[row][0].set_ylabel(
            f"{'checkmark' if ltype == 'correct' else 'x'} true={true_label}",
            fontsize=9, rotation=0, labelpad=55, va="center")
        axes[row][0].axis("off")
        axes[row][1].imshow(overlay)
        axes[row][1].set_title(
            f"pred={pred}  conf={conf:.1%}", fontsize=8,
            color="green" if pred == true_label else "red")
        axes[row][1].axis("off")
        print(f"    [{ltype}] idx={idx} true={true_label} pred={pred} conf={conf:.1%}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GradCAM for all EfficientNet-B4 Flowers102 WandB runs"
    )
    parser.add_argument("--data-root",    type=str,
                        default="/ocean/projects/cis260045p/shared/data")
    parser.add_argument("--output-dir",   type=str,
                        default="/ocean/projects/cis260045p/labebe/gradcam_outputs")
    parser.add_argument("--cache-dir",    type=str,
                        default="./wandb_checkpoints",
                        help="Local folder to cache downloaded checkpoints")
    parser.add_argument("--run-name",     type=str, default=None,
                        help="Process only this run (must match name exactly)")
    parser.add_argument("--no-cuda",      action="store_true", default=False)
    parser.add_argument("--n-correct",    type=int, default=4)
    parser.add_argument("--n-incorrect",  type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir,  exist_ok=True)

    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    _, val_transform, _, _ = build_transforms_efficientnet_b4()
    dataset_builder = get_dataset_builder("flowers102")
    _, val_dataset, _ = dataset_builder(
        data_root=args.data_root,
        train_transform=val_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        download=False,
    )
    print(f"Val set: {len(val_dataset)} images\n")

    # ── WandB API ─────────────────────────────────────────────────────────────
    api = wandb.Api()

    # ── Filter runs if requested ──────────────────────────────────────────────
    runs_to_process = ALL_RUNS
    if args.run_name:
        runs_to_process = [r for r in ALL_RUNS if r[0] == args.run_name]
        if not runs_to_process:
            print(f"ERROR: '{args.run_name}' not found. Available run names:")
            for r in ALL_RUNS:
                print(f"  {r[0]}")
            sys.exit(1)

    # ── Process each run ──────────────────────────────────────────────────────
    failed = []
    for run_name, artifact_name, model_type in runs_to_process:
        print(f"\n{'='*60}")
        print(f"Run: {run_name}  [{model_type}]")
        print(f"{'='*60}")
        try:
            ckpt_path = download_artifact(api, artifact_name, args.cache_dir)
            model     = load_model(ckpt_path, model_type, device)
            safe_name = (run_name.replace(" ", "_").replace("/", "_")
                                 .replace("(", "").replace(")", "")
                                 .replace(".", ""))
            out_path  = os.path.join(args.output_dir, f"gradcam_{safe_name}.png")
            make_figure(model, val_dataset, device, out_path, run_name,
                        n_correct=args.n_correct,
                        n_incorrect=args.n_incorrect)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR on '{run_name}': {e}")
            failed.append(run_name)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    success = len(runs_to_process) - len(failed)
    print(f"Done. {success}/{len(runs_to_process)} runs completed.")
    if failed:
        print(f"Failed: {failed}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
