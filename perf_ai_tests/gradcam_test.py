
"""
Quick GradCAM test script — single model, no comparison.
Use this to verify GradCAM is working before running the full comparison.
 
Usage:
------
# Test with the shared Flowers102 perforated checkpoint
python gradcam_test.py \
    --checkpoint /ocean/projects/cis260045p/shared/outputs_flowers_efficientnet/efficientnet_b5_flowers102_best.pt \
    --dataset flowers102 \
    --data-root /ocean/projects/cis260045p/shared/data \
    --output-dir /ocean/projects/cis260045p/labebe/gradcam_test_outputs \
    --model-type perforated
 
# If that fails, try the PAI rank0 best_model.pt
python gradcam_test.py \
    --checkpoint /ocean/projects/cis260045p/shared/outputs_flowers_efficientnet/efficientnet_b5_flowers102_pai_rank0/best_model.pt \
    --dataset flowers102 \
    --data-root /ocean/projects/cis260045p/shared/data \
    --output-dir /ocean/projects/cis260045p/labebe/gradcam_test_outputs \
    --model-type perforated
"""
 
from __future__ import annotations
import argparse
import os
import sys
from typing import List, Tuple
 
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
    print("ERROR: pip install grad-cam")
    sys.exit(1)
 
try:
    from models import EfficientNetB5PAI, build_transforms, efficientnet_b5_flowers102
    from data.registry import get_dataset_builder
except ImportError as e:
    print(f"ERROR: {e}")
    print("Run from ~/idls26_perfAI/ with conda env activated.")
    sys.exit(1)
 
 
DATASET_NUM_CLASSES = {
    "flowers102": 102,
    "pets":       37,
    "food101":    101,
}
 
 
def load_model(checkpoint_path: str, num_classes: int, model_type: str,
               device: torch.device) -> torch.nn.Module:
    """
    model_type = 'perforated' → loads EfficientNetB5PAI wrapper
    model_type = 'baseline'   → loads plain efficientnet_b5_flowers102
    """
    print(f"Loading {model_type} model from: {checkpoint_path}")
 
    base = efficientnet_b5_flowers102(num_classes=num_classes, finetune_backbone=False)
 
    if model_type == "perforated":
        model = EfficientNetB5PAI(base)
    else:
        model = base
 
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
 
    model_keys = set(model.state_dict().keys())
    filtered   = {k: v for k, v in state_dict.items() if k in model_keys}
    missing    = model_keys - set(filtered.keys())
 
    if missing:
        print(f"  WARNING: {len(missing)} missing keys — loading with strict=False")
        model.load_state_dict(filtered, strict=False)
    else:
        model.load_state_dict(filtered, strict=True)
        print(f"  Loaded cleanly — all {len(filtered)} keys matched.")
 
    model.eval()
    model.to(device)
    return model
 
 
def get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Works for both plain EfficientNet and EfficientNetB5PAI."""
    if hasattr(model, "features"):
        layer = model.features[-1]
        print(f"  GradCAM target: model.features[-1] = {type(layer).__name__}")
        return layer
    raise AttributeError("Cannot find 'features' on model.")
 
 
@torch.no_grad()
def collect_examples(model, dataset, device, n_correct=4, n_incorrect=3):
    correct, incorrect = [], []
    for idx in range(len(dataset)):
        if len(correct) >= n_correct and len(incorrect) >= n_incorrect:
            break
        img, label = dataset[idx]
        inp  = img.unsqueeze(0).to(device)
        pred = model(inp).argmax(dim=1).item()
        if pred == label and len(correct) < n_correct:
            correct.append(idx)
        elif pred != label and len(incorrect) < n_incorrect:
            incorrect.append(idx)
    print(f"  Found {len(correct)} correct, {len(incorrect)} incorrect examples.")
    return correct, incorrect
 
 
def run_gradcam(model, target_layer, img_tensor, target_class, device):
    cam = GradCAM(model=model, target_layers=[target_layer])
    inp = img_tensor.to(device)
    inp.requires_grad_(True)
    
    with torch.enable_grad():
        grayscale = cam(input_tensor=inp,
                        targets=[ClassifierOutputTarget(target_class)])[0]

    rgb = inp.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
    rgb = np.clip(
        np.array([0.229, 0.224, 0.225]) * rgb + np.array([0.485, 0.456, 0.406]),
        0, 1
    ).astype(np.float32)

    overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

    with torch.no_grad():
        probs = F.softmax(model(inp), dim=1)
        pred  = probs.argmax(dim=1).item()
        conf  = probs[0, pred].item()

    return overlay, rgb, pred, conf
 
 
def make_figure(model, target_layer, dataset, device, output_path,
                n_correct=4, n_incorrect=3):
    correct_idx, incorrect_idx = collect_examples(
        model, dataset, device, n_correct, n_incorrect
    )
    all_idx   = correct_idx + incorrect_idx
    all_types = ["correct"] * len(correct_idx) + ["incorrect"] * len(incorrect_idx)
 
    n = len(all_idx)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n), squeeze=False)
    fig.suptitle("GradCAM Test — EfficientNet-B5 Flowers102",
                 fontsize=13, fontweight="bold")
 
    axes[0][0].set_title("Original Image",    fontsize=11, fontweight="bold")
    axes[0][1].set_title("GradCAM Heatmap",   fontsize=11, fontweight="bold")
 
    for row, (idx, ltype) in enumerate(zip(all_idx, all_types)):
        img_tensor, true_label = dataset[idx]
        inp = img_tensor.unsqueeze(0)
 
        overlay, rgb, pred, conf = run_gradcam(
            model, target_layer, inp, true_label, device
        )
 
        axes[row][0].imshow(rgb)
        axes[row][0].set_ylabel(
            f"{'✓' if ltype == 'correct' else '✗'} true={true_label}",
            fontsize=9, rotation=0, labelpad=50, va="center"
        )
        axes[row][0].axis("off")
 
        axes[row][1].imshow(overlay)
        axes[row][1].set_title(
            f"pred={pred}  conf={conf:.1%}",
            fontsize=8,
            color="green" if pred == true_label else "red"
        )
        axes[row][1].axis("off")
 
        print(f"  [{ltype}] idx={idx} true={true_label} pred={pred} conf={conf:.1%}")
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {output_path}")
 
 
def main():
    parser = argparse.ArgumentParser(description="Quick GradCAM test — single model")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--dataset",     type=str, default="flowers102",
                        choices=["flowers102", "pets", "food101"])
    parser.add_argument("--model-type",  type=str, default="perforated",
                        choices=["perforated", "baseline"],
                        help="perforated=EfficientNetB5PAI, baseline=plain EfficientNet")
    parser.add_argument("--data-root",   type=str,
                        default="/ocean/projects/cis260045p/shared/data")
    parser.add_argument("--output-dir",  type=str,
                        default="/ocean/projects/cis260045p/labebe/gradcam_test_outputs")
    parser.add_argument("--no-cuda",     action="store_true", default=False)
    parser.add_argument("--n-correct",   type=int, default=4)
    parser.add_argument("--n-incorrect", type=int, default=3)
    args = parser.parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")
 
    num_classes = DATASET_NUM_CLASSES[args.dataset]
    model       = load_model(args.checkpoint, num_classes, args.model_type, device)
    target_layer = get_target_layer(model)
 
    _, val_transform, _, _ = build_transforms()
    dataset_builder = get_dataset_builder(args.dataset)
    _, val_dataset, _ = dataset_builder(
        data_root=args.data_root,
        train_transform=val_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        download=False,
    )
    print(f"Val set: {len(val_dataset)} images")
 
    out_path = os.path.join(
        args.output_dir,
        f"gradcam_test_{args.dataset}_{args.model_type}.png"
    )
    make_figure(model, target_layer, val_dataset, device, out_path,
                args.n_correct, args.n_incorrect)
 
 
if __name__ == "__main__":
    main()
