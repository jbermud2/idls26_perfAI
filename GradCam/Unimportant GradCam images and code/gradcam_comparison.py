"""
GradCAM Comparison: EfficientNet B4 Baseline (0 dendrites) vs 1 Dendrite
=========================================================================
Handles the difference in checkpoint formats between:
  - main_baseline.py  → saves as {"model_state_dict": ..., "epoch": ...}
  - main.py (PAI)     → saves via UPA.save_system(), PAI-wrapped model

Requirements:
    pip install wandb torch torchvision grad-cam Pillow matplotlib timm

    You also need the PerforatedAI package installed in your environment
    (same conda env used on PSC):
        conda activate /ocean/projects/cis260045p/shared/perf_ai

Usage (on PSC, inside your interactive GPU session):
    python gradcam_comparison.py \
        --data_root /ocean/projects/cis260045p/shared/data \
        --output_dir ./gradcam_outputs

    # Skip WandB download if you already have checkpoints locally:
    python gradcam_comparison.py \
        --data_root /ocean/projects/cis260045p/shared/data \
        --baseline_ckpt ./artifacts_efficientnet_b4_flowers102_baseline/best_model.pt \
        --dendrite_pai_dir ./artifacts_efficientnet_b4_flowers102

Notes on checkpoint paths:
  - Baseline best model:  artifacts_efficientnet_b4_flowers102_baseline/best_model.pt
  - Dendrite PAI dir:     artifacts_efficientnet_b4_flowers102/
    (UPA.load_system expects the directory name, not a specific file)
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102

import wandb
import timm

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# PerforatedAI — must be installed in your active conda env
try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    PAI_AVAILABLE = True
except ImportError:
    PAI_AVAILABLE = False
    print("[WARNING] PerforatedAI not found. Dendrite model loading will fall back to "
          "plain state_dict loading, which may not work correctly.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY  = "PerforatedAI_IDL"
WANDB_PROJECT = "efficientnet_b4_flowers102"

# Artifact paths from WandB (as seen in your screenshots)
ARTIFACT_BASELINE = (
    "PerforatedAI_IDL/efficientnet_b4_flowers102/"
    "artifacts_efficientnet_b4_flowers102_baseline-best:v0"
)
ARTIFACT_DENDRITE = (
    "PerforatedAI_IDL/efficientnet_b4_flowers102/"
    "artifacts_efficientnet_b4_flowers102-best:v7"
)

# PAI save name used during training (from args.pai_save_name in main.py)
PAI_SAVE_NAME = "artifacts_efficientnet_b4_flowers102"

NUM_CLASSES = 102
IMG_SIZE    = 380      # EfficientNet-B4 native resolution
NUM_SAMPLES = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# FLOWER CLASS NAMES
# ─────────────────────────────────────────────────────────────────────────────
FLOWER_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barberton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "pink quill", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]


# ─────────────────────────────────────────────────────────────────────────────
# WANDB ARTIFACT DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_artifact(artifact_path: str, download_dir: str) -> str:
    print(f"\nDownloading WandB artifact: {artifact_path}")
    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")
    local_dir = artifact.download(root=download_dir)
    print(f"  Saved to: {local_dir}")
    return local_dir


def find_file(directory: str, ext: str) -> str | None:
    """Recursively find first file with given extension."""
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(ext):
                return os.path.join(root, f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDING
# The baseline and dendrite models were built differently during training.
# We must replicate each path exactly.
# ─────────────────────────────────────────────────────────────────────────────
def build_plain_efficientnet_b4(num_classes: int = 102) -> nn.Module:
    """
    Matches main_baseline.py — checkpoint keys use torchvision format:
    features.0.0.weight, features.0.1.weight, etc.
    Must use torchvision (not timm) to match the saved state dict.
    """
    import torchvision.models as tv
    model = tv.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_efficientnet_b4_with_pre_fc(num_classes: int = 102) -> nn.Module:
    """
    Matches the DENDRITE training architecture.
    Inspecting the safetensors checkpoint revealed two linear layers:
      - pre_fc:        1792 -> 1792  (PAI dendrite target)
      - classifier_fc: 1792 -> 102   (final output)
    The training wrap_model() adds a hidden pre_fc layer before the classifier.
    """
    import torchvision.models as tv
    model = tv.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features  # 1792 for B4

    # Replace classifier with dropout + pre_fc linear
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, in_features),  # pre_fc: 1792->1792
    )
    # Add classifier_fc as top-level attribute to match checkpoint key names
    model.classifier_fc = nn.Linear(in_features, num_classes)

    def new_forward(x):
        x = model.features(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.classifier(x)      # dropout + pre_fc
        x = model.classifier_fc(x)   # final 102-class output
        return x

    model.forward = new_forward
    return model


def load_baseline_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    main_baseline.py saves best_model.pt as:
        {
            "epoch": ...,
            "model_state_dict": model.state_dict(),   <-- key we need
            "optimizer_state_dict": ...,
            "scheduler_state_dict": ...,
        }
    """
    print(f"\nLoading BASELINE checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # Assume the file IS the state dict
        state_dict = ckpt

    # Strip DataParallel / DDP "module." prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    model.eval()
    model.to(DEVICE)
    print("  Baseline checkpoint loaded successfully.")
    return model


def load_dendrite_model(pai_dir: str, num_classes: int = 102) -> nn.Module:
    """
    main.py (PAI) builds the dendrite model as:
        base_model = model_config.build_model(num_classes=..., finetune_backbone=True)
        model = model_config.wrap_model(base_model)
        configure_pai(args, model)
        model = UPA.perforate_model(model, save_name=pai_save_name)
        # ... training ...
        UPA.save_system(model, pai_save_name, "best_model")

    So we must:
        1. Build the base model
        2. wrap_model (timm EfficientNet-B4 has no special wrapper — it's identity)
        3. perforate_model with the same save_name
        4. UPA.load_system(model, pai_save_name, "best_model", switch=True)

    The 'pai_dir' argument should be the folder that was used as pai_save_name
    during training (i.e. "artifacts_efficientnet_b4_flowers102").
    WandB downloads the best_model.pt file; we just need it to be in that dir.
    """
    if not PAI_AVAILABLE:
        raise RuntimeError(
            "PerforatedAI is required to load the dendrite model. "
            "Activate your PSC conda env: conda activate /ocean/projects/cis260045p/shared/perf_ai"
        )

    print(f"\nBuilding dendrite model from PAI dir: {pai_dir}")

    import torchvision.models as tv
    from safetensors import safe_open

    # 1. Build model with the correct pre_fc architecture matching training
    base_model = build_efficientnet_b4_with_pre_fc(num_classes)

    # 2. Configure GPA — target is classifier[1] which is the pre_fc layer (index 1 in Sequential)
    GPA.pc.set_verbose(False)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_module_names_to_perforate([])

    convert_module_id = ".classifier.1"   # pre_fc is classifier[1] in the Sequential
    GPA.pc.set_module_ids_to_perforate([convert_module_id])

    ids_to_track = [
        f".{name}" for name, module in base_model.named_modules()
        if len(list(module.children())) == 0
        and name != convert_module_id.lstrip(".")
    ]
    GPA.pc.set_module_ids_to_track(ids_to_track)

    if hasattr(GPA.pc, "set_weight_decay_accepted"):
        GPA.pc.set_weight_decay_accepted(True)

    GPA.pc.set_improvement_threshold([0.001, 0.0001, 0])
    GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
    GPA.pc.set_fixed_switch_num(25)
    GPA.pc.set_first_fixed_switch_num(25)
    GPA.pc.set_candidate_weight_initialization_multiplier(0.1)
    GPA.pc.set_max_dendrites(1)

    import torch.nn.functional as F
    GPA.pc.set_pai_forward_function(F.relu)
    if hasattr(GPA.pc, "set_perforated_backpropagation"):
        GPA.pc.set_perforated_backpropagation(False)

    print("  Initializing PAI scaffold...")
    model = UPA.initialize_pai(
        base_model,
        doing_pai=True,
        save_name=pai_dir,
        making_graphs=False,
        maximizing_score=True,
        num_classes=num_classes,
    )

    # 3. Load weights from safetensors.
    #    Checkpoint keys:
    #      pre_fc.*        -> classifier.1.*   (PAI dendrite layer)
    #      classifier_fc.* -> classifier_fc.*  (final output, top-level attr)
    #      features.*      -> features.*       (backbone, no remapping needed)
    print("  Loading weights from safetensors checkpoint...")
    ckpt_path = os.path.join(pai_dir, "best_model.pt")

    tensors = {}
    with safe_open(ckpt_path, framework="pt", device=str(DEVICE)) as f:
        for key in f.keys():
            # Skip PAI internal tracking buffers that aren't model weights
            if key in ("tracker_string",):
                continue
            new_key = key.replace("pre_fc.", "classifier.1.")
            tensors[new_key] = f.get_tensor(key)
    
    # tensors = {}
    # with safe_open(ckpt_path, framework="pt", device=str(DEVICE)) as f:
    #     for key in f.keys():
    #         new_key = key.replace("pre_fc.", "classifier.1.")
    #         tensors[new_key] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(tensors, strict=False)
    real_missing = [k for k in missing if not any(x in k for x in
                    ["num_batches", "tracker", "this_node", "this_output", "dendrites_to_top"])]
    real_unexpected = [k for k in unexpected if not any(x in k for x in
                       ["tracker_string", "this_node", "this_output", "dendrites_to_top"])]
    if real_missing:
        print(f"  [WARN] Missing keys ({len(real_missing)}): {real_missing[:5]}")
    if real_unexpected:
        print(f"  [WARN] Unexpected keys ({len(real_unexpected)}): {real_unexpected[:5]}")

    model.eval()
    model.to(DEVICE)
    print("  Dendrite model loaded successfully from safetensors.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# GRADCAM TARGET LAYER RESOLUTION
# After UPA.perforate_model(), the internal structure may shift.
# We probe the model to find the last convolutional layer dynamically.
# ─────────────────────────────────────────────────────────────────────────────
def get_last_conv_layer(model: nn.Module) -> list:
    """
    Dynamically find the last Conv2d in the model.
    This is robust to PAI restructuring the layer tree.

    For vanilla EfficientNet-B4 (timm), this will be inside blocks[-1][-1].
    For the PAI-perforated model, the dendrite node is inserted after this layer,
    so the last conv is still the richest feature representation before the dendrite.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise RuntimeError("No Conv2d layers found in model.")

    print(f"  GradCAM target layer: {type(last_conv).__name__} "
          f"(out_channels={last_conv.out_channels})")
    return [last_conv]


def get_named_target_layer(model: nn.Module, name: str) -> list:
    """
    Alternative: look up a layer by its named path.
    Use this if you know the exact layer name after PAI wrapping.
    Example: name = "blocks.6.1.conv_pwl"
    """
    for n, m in model.named_modules():
        if n == name:
            print(f"  GradCAM target layer (by name): {n}")
            return [m]
    raise ValueError(f"Layer '{name}' not found in model. "
                     f"Available conv layers:\n" +
                     "\n".join(n for n, _ in model.named_modules() if "conv" in n.lower()))


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return inference_transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)


def tensor_to_rgb_array(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.squeeze(0).cpu() * std + mean
    return img.permute(1, 2, 0).numpy().clip(0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# RUN GRADCAM
# ─────────────────────────────────────────────────────────────────────────────
def run_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    use_plusplus: bool = False,
) -> np.ndarray:
    target_layers = get_last_conv_layer(model)
    CAMClass = GradCAMPlusPlus if use_plusplus else GradCAM

    # PAI-perforated models may have custom autograd functions for dendrite nodes.
    # Setting use_cuda=True here ensures gradients flow correctly on GPU.
    with CAMClass(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD TEST SAMPLES FROM FLOWERS102
# ─────────────────────────────────────────────────────────────────────────────
def load_test_samples(data_root: str, num_samples: int):
    dataset = Flowers102(
        root=data_root,
        split="test",
        transform=None,
        download=True,
    )
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    samples = []
    for idx in indices:
        img, label = dataset[idx]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        samples.append((img, label))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPARISON FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def build_comparison_figure(
    samples,
    baseline_model: nn.Module,
    dendrite_model: nn.Module,
    output_path: str = "gradcam_comparison.png",
    use_plusplus: bool = False,
):
    n = len(samples)
    cam_method = "GradCAM++" if use_plusplus else "GradCAM"

    #fig = plt.figure(figsize=(15, 5 * n))
    fig = plt.figure(figsize=(15, 6 * n))
    fig.suptitle(
        f"EfficientNet-B4 {cam_method}: Baseline (0 Dendrites) vs 1 Dendrite\n"
        f"Oxford Flowers 102  |  Baseline: 92.57% Top-1  |  1 Dendrite: 95.09% Top-1",
        fontsize=14, fontweight="bold", y=1.01,
    )

    #gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.45, wspace=0.08)
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.6, wspace=0.15)

    col_titles = [
        "Original Image\n(Ground Truth)",
        "Baseline — 0 Dendrites\n17.7M params  |  92.57% Top-1",
        "1 Dendrite (PAI)\n24.2M params  |  95.09% Top-1",
    ]

    for col_i, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col_i])
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
        ax.axis("off")

    for row_i, (pil_img, true_label) in enumerate(samples):
        input_tensor = pil_to_tensor(pil_img)
        rgb_arr = tensor_to_rgb_array(input_tensor)
        class_name = FLOWER_CLASSES[true_label] if true_label < len(FLOWER_CLASSES) else f"class {true_label}"

        with torch.no_grad():
            b_logits = baseline_model(input_tensor)
            d_logits = dendrite_model(input_tensor)

        b_pred = b_logits.argmax(1).item()
        d_pred = d_logits.argmax(1).item()
        b_conf = torch.softmax(b_logits, 1)[0, b_pred].item()
        d_conf = torch.softmax(d_logits, 1)[0, d_pred].item()
        b_name = FLOWER_CLASSES[b_pred] if b_pred < len(FLOWER_CLASSES) else str(b_pred)
        d_name = FLOWER_CLASSES[d_pred] if d_pred < len(FLOWER_CLASSES) else str(d_pred)

        # GradCAM targeted at TRUE class — shows what each model uses
        # to recognize the correct flower, not just what it happens to predict
        cam_b = run_gradcam(baseline_model, input_tensor, true_label, use_plusplus)
        cam_d = run_gradcam(dendrite_model,  input_tensor, true_label, use_plusplus)

        overlay_b = show_cam_on_image(rgb_arr, cam_b, use_rgb=True)
        overlay_d = show_cam_on_image(rgb_arr, cam_d, use_rgb=True)

        ax0 = fig.add_subplot(gs[row_i, 0])
        ax0.imshow(pil_img.resize((IMG_SIZE, IMG_SIZE)))
        #ax0.set_title(f"True: {class_name}", fontsize=9)
        ax0.set_title(f"True: {class_name}", fontsize=11, pad=8)
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[row_i, 1])
        ax1.imshow(overlay_b)
        marker = "✓" if b_pred == true_label else "✗"
        ax1.set_title(
            f"Pred: {b_name} {marker}\nConf: {b_conf:.1%}",
            fontsize=10,
            color="green" if b_pred == true_label else "red",
        )
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[row_i, 2])
        ax2.imshow(overlay_d)
        marker = "✓" if d_pred == true_label else "✗"
        ax2.set_title(
            f"Pred: {d_name} {marker}\nConf: {d_conf:.1%}",
            fontsize=10,
            color="green" if d_pred == true_label else "red",
        )
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison figure → {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE CASE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def find_interesting_cases(data_root, baseline_model, dendrite_model, max_cases=6):
    """
    Scan test set for:
      - Dendrite wins: baseline wrong, dendrite right
      - Both wrong but disagree: different misclassifications (shows divergent attention)
    """
    dataset = Flowers102(root=data_root, split="test", transform=None, download=False)
    dendrite_wins, shared_failures = [], []

    print("\nScanning test set for interesting cases (up to 500 images)...")
    for idx in range(min(500, len(dataset))):
        img, label = dataset[idx]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        t = pil_to_tensor(img)
        with torch.no_grad():
            b_pred = baseline_model(t).argmax(1).item()
            d_pred = dendrite_model(t).argmax(1).item()

        if b_pred != label and d_pred == label:
            dendrite_wins.append((img, label, b_pred, d_pred))
        elif b_pred != label and d_pred != label and b_pred != d_pred:
            shared_failures.append((img, label, b_pred, d_pred))

        target_each = max_cases // 2
        if len(dendrite_wins) >= target_each and len(shared_failures) >= target_each:
            break

    print(f"  Found {len(dendrite_wins)} dendrite-wins, {len(shared_failures)} shared-failures")
    return dendrite_wins[:max_cases // 2], shared_failures[:max_cases // 2]


def build_failure_figure(dendrite_wins, shared_failures, baseline_model, dendrite_model,
                         output_path="gradcam_failure_analysis.png"):
    all_cases = dendrite_wins + shared_failures
    row_labels = (["Dendrite Wins"] * len(dendrite_wins) +
                  ["Both Wrong (Disagree)"] * len(shared_failures))

    if not all_cases:
        print("No interesting cases found in the scanned range.")
        return

    n = len(all_cases)
    fig, axes = plt.subplots(n, 3, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Failure Case Analysis: Baseline vs 1 Dendrite\n"
        "GradCAM targeted at ground-truth class",
        fontsize=14, fontweight="bold"
    )

    for i, ((img, true_label, b_pred, d_pred), row_label) in enumerate(zip(all_cases, row_labels)):
        t = pil_to_tensor(img)
        rgb = tensor_to_rgb_array(t)

        cam_b = run_gradcam(baseline_model, t, true_label)
        cam_d = run_gradcam(dendrite_model,  t, true_label)

        overlay_b = show_cam_on_image(rgb, cam_b, use_rgb=True)
        overlay_d = show_cam_on_image(rgb, cam_d, use_rgb=True)

        true_name = FLOWER_CLASSES[true_label] if true_label < len(FLOWER_CLASSES) else str(true_label)
        b_name    = FLOWER_CLASSES[b_pred]     if b_pred     < len(FLOWER_CLASSES) else str(b_pred)
        d_name    = FLOWER_CLASSES[d_pred]     if d_pred     < len(FLOWER_CLASSES) else str(d_pred)

        axes[i][0].imshow(img.resize((IMG_SIZE, IMG_SIZE)))
        axes[i][0].set_title(f"[{row_label}]\nTrue: {true_name}", fontsize=10)
        axes[i][0].axis("off")

        axes[i][1].imshow(overlay_b)
        axes[i][1].set_title(
            f"Baseline pred: {b_name}",
            fontsize=10, color="red" if b_pred != true_label else "green"
        )
        axes[i][1].axis("off")

        axes[i][2].imshow(overlay_d)
        axes[i][2].set_title(
            f"Dendrite pred: {d_name}",
            fontsize=10, color="green" if d_pred == true_label else "red"
        )
        axes[i][2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved failure analysis → {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ARGS + MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",        default="/ocean/projects/cis260045p/shared/data")
    p.add_argument("--artifact_dir",     default="./wandb_artifacts")
    p.add_argument("--output_dir",       default="./gradcam_outputs")
    p.add_argument("--num_samples",      type=int, default=NUM_SAMPLES)
    p.add_argument("--use_plusplus",     action="store_true")
    p.add_argument("--skip_failure",     action="store_true")
    p.add_argument("--baseline_ckpt",    default=None,
                   help="Path to baseline best_model.pt (skips WandB download)")
    p.add_argument("--dendrite_pai_dir", default=None,
                   help="Path to the full PAI artifacts directory for the dendrite model "
                        "(e.g. ./artifacts_efficientnet_b4_flowers102). "
                        "Must contain best_model.pt AND PAI config files.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)

    # ── 1. Resolve baseline checkpoint ───────────────────────────────────────
    if args.baseline_ckpt:
        baseline_ckpt = args.baseline_ckpt
    else:
        baseline_art_dir = download_artifact(
            ARTIFACT_BASELINE,
            os.path.join(args.artifact_dir, "baseline"),
        )
        baseline_ckpt = find_file(baseline_art_dir, ".pt")
        if not baseline_ckpt:
            raise FileNotFoundError(f"No .pt file found in {baseline_art_dir}")

    # ── 2. Resolve PAI directory for dendrite model ───────────────────────────
    if args.dendrite_pai_dir:
        pai_dir = args.dendrite_pai_dir
    else:
        dendrite_art_dir = download_artifact(
            ARTIFACT_DENDRITE,
            os.path.join(args.artifact_dir, PAI_SAVE_NAME),
        )
        pai_dir = dendrite_art_dir
        print(
            f"\n[NOTE] UPA.load_system needs the full PAI artifacts folder, not just best_model.pt.\n"
            f"If loading fails, copy the full folder from PSC and re-run with --dendrite_pai_dir:\n"
            f"  scp -r <user>@bridges2.psc.edu:~/idls26_perfAI/{PAI_SAVE_NAME}/ ./\n"
            f"  python gradcam_comparison.py --dendrite_pai_dir ./{PAI_SAVE_NAME} ..."
        )

    # ── 3. Build and load models ──────────────────────────────────────────────
    print("\n=== Loading BASELINE model (0 dendrites) ===")
    baseline_model = build_plain_efficientnet_b4(NUM_CLASSES)
    baseline_model = load_baseline_checkpoint(baseline_model, baseline_ckpt)

    print("\n=== Loading DENDRITE model (1 dendrite, PAI) ===")
    dendrite_model = load_dendrite_model(pai_dir, NUM_CLASSES)

    # ── 4. Load test samples ──────────────────────────────────────────────────
    print(f"\nLoading {args.num_samples} test samples from Flowers102...")
    samples = load_test_samples(args.data_root, args.num_samples)

    # ── 5. Main comparison figure ─────────────────────────────────────────────
    out_main = os.path.join(args.output_dir, "gradcam_comparison.png")
    build_comparison_figure(
        samples, baseline_model, dendrite_model,
        output_path=out_main,
        use_plusplus=args.use_plusplus,
    )

    # ── 6. Failure case analysis ──────────────────────────────────────────────
    if not args.skip_failure:
        wins, failures = find_interesting_cases(
            args.data_root, baseline_model, dendrite_model
        )
        out_fail = os.path.join(args.output_dir, "gradcam_failure_analysis.png")
        build_failure_figure(wins, failures, baseline_model, dendrite_model, out_fail)

    print(f"\nDone! All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
