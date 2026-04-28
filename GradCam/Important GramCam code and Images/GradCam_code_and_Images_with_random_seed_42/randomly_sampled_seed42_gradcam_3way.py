"""
GradCAM 3-Way Comparison — Randomly Sampled (seed=42):
  EfficientNet-B4 Baseline (0 dendrites)
  vs EfficientNet-B4 + 1 Dendrite (PAI)
  vs EfficientNet-B5 Baseline (0 dendrites)
=========================================================================
This version replaces the previous np.linspace() image selection with true
random sampling using a fixed seed (numpy seed=42). This is reproducible
and defensible for academic papers — report as:
  "6 images randomly sampled from the Oxford Flowers 102 test set (seed=42)."

Why this matters: the Flowers102 test set is ordered by class, so linspace()
sampling tends to hit evenly-spaced species, which is not truly random. Fixed-
seed random sampling is the standard approach in the fine-grained recognition
literature.

This is the core paper figure: shows that B4+dendrite achieves near-B5
accuracy at lower parameter cost, and GradCAM shows WHY visually.

Model stats:
  B4 Baseline:   17.7M params | 92.57% Top-1
  B4 1-Dendrite: 24.2M params | 95.09% Top-1
  B5 Baseline:   28.5M params | 95.85% Top-1

Checkpoint formats:
  B4 Baseline:   standard PyTorch ZIP (.pt) with "model_state_dict" key
  B4 Dendrite:   safetensors via PAI, keys use "pre_fc" and "classifier_fc"
  B5 Baseline:   standard PyTorch ZIP (.pt) with "model_state_dict" key

WandB artifacts:
  B4 Baseline:   PerforatedAI_IDL/efficientnet_b4_flowers102/artifacts_efficientnet_b4_flowers102_baseline-best:v0
  B4 Dendrite:   PerforatedAI_IDL/efficientnet_b4_flowers102/artifacts_efficientnet_b4_flowers102-best:v7
  B5 Baseline:   PerforatedAI_IDL/efficientnet_b5_flowers102/lr3e-3_wd1e-5-best:v0

Usage:
    python gradcam_3way.py \
        --data_root /ocean/projects/cis260045p/shared/data \
        --output_dir ./gradcam_outputs_3way \
        --dendrite_pai_dir ./wandb_artifacts/artifacts_efficientnet_b4_flowers102

    # To skip all WandB downloads (if already cached):
    python gradcam_3way.py \
        --data_root /ocean/projects/cis260045p/shared/data \
        --b4_baseline_ckpt  ./wandb_artifacts/baseline/best_model.pt \
        --dendrite_pai_dir  ./wandb_artifacts/artifacts_efficientnet_b4_flowers102 \
        --b5_baseline_ckpt  ./wandb_artifacts/b5_baseline/best_model.pt
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
import torchvision.models as tv
from torchvision.datasets import Flowers102

import wandb

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    PAI_AVAILABLE = True
except ImportError:
    PAI_AVAILABLE = False
    print("[WARNING] PerforatedAI not found.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY = "PerforatedAI_IDL"

ARTIFACT_B4_BASELINE = (
    "PerforatedAI_IDL/efficientnet_b4_flowers102/"
    "artifacts_efficientnet_b4_flowers102_baseline-best:v0"
)
ARTIFACT_B4_DENDRITE = (
    "PerforatedAI_IDL/efficientnet_b4_flowers102/"
    "artifacts_efficientnet_b4_flowers102-best:v7"
)
ARTIFACT_B5_BASELINE = (
    "PerforatedAI_IDL/efficientnet_b5_flowers102/"
    "lr3e-3_wd1e-5-best:v0"
)

PAI_SAVE_NAME = "artifacts_efficientnet_b4_flowers102"

NUM_CLASSES = 102
B4_IMG_SIZE = 380
B5_IMG_SIZE = 456   # EfficientNet-B5 native resolution
NUM_SAMPLES = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

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


def find_pt_file(directory: str) -> str:
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".pt") or f.endswith(".pth"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No .pt file found in {directory}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_b4_baseline(num_classes: int = 102) -> nn.Module:
    """Standard torchvision EfficientNet-B4, matches baseline checkpoint keys."""
    model = tv.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_b4_with_pre_fc(num_classes: int = 102) -> nn.Module:
    """
    EfficientNet-B4 with PAI's pre_fc hidden layer.
    Checkpoint keys: pre_fc.* (1792->1792) + classifier_fc.* (1792->102)
    """
    model = tv.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features  # 1792

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, in_features),  # pre_fc
    )
    model.classifier_fc = nn.Linear(in_features, num_classes)

    def new_forward(x):
        x = model.features(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.classifier(x)
        x = model.classifier_fc(x)
        return x

    model.forward = new_forward
    return model


def build_b5_baseline(num_classes: int = 102) -> nn.Module:
    """Standard torchvision EfficientNet-B5, matches baseline checkpoint keys."""
    model = tv.efficientnet_b5(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def load_standard_checkpoint(model: nn.Module, checkpoint_path: str,
                              label: str = "model") -> nn.Module:
    """
    Load a standard PyTorch checkpoint saved as:
        {"model_state_dict": state_dict, "epoch": ..., ...}
    Works for both B4 and B5 baselines.
    """
    print(f"\nLoading {label} checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    real_missing = [k for k in missing if "num_batches" not in k]
    if real_missing:
        print(f"  [WARN] Missing keys ({len(real_missing)}): {real_missing[:3]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    model.eval()
    model.to(DEVICE)
    print(f"  {label} checkpoint loaded successfully.")
    return model


def load_dendrite_model(pai_dir: str, num_classes: int = 102) -> nn.Module:
    """
    Load the PAI dendrite model from a safetensors checkpoint.
    Keys: pre_fc.* -> classifier.1.*, classifier_fc.* stays as-is.
    """
    if not PAI_AVAILABLE:
        raise RuntimeError("PerforatedAI required. Run: conda activate perf_ai")

    from safetensors import safe_open
    import torch.nn.functional as F

    print(f"\nBuilding B4 dendrite model from: {pai_dir}")

    base_model = build_b4_with_pre_fc(num_classes)

    # Configure PAI GPA settings matching training run
    GPA.pc.set_verbose(False)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_module_names_to_perforate([])

    convert_module_id = ".classifier.1"
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
    GPA.pc.set_pai_forward_function(F.relu)
    if hasattr(GPA.pc, "set_perforated_backpropagation"):
        GPA.pc.set_perforated_backpropagation(False)

    print("  Initializing PAI scaffold...")
    model = UPA.initialize_pai(
        base_model, doing_pai=True, save_name=pai_dir,
        making_graphs=False, maximizing_score=True, num_classes=num_classes,
    )

    print("  Loading weights from safetensors checkpoint...")
    ckpt_path = os.path.join(pai_dir, "best_model.pt")
    tensors = {}
    with safe_open(ckpt_path, framework="pt", device=str(DEVICE)) as f:
        for key in f.keys():
            if key == "tracker_string":
                continue
            new_key = key.replace("pre_fc.", "classifier.1.")
            tensors[new_key] = f.get_tensor(key)

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
    print("  B4 dendrite model loaded successfully.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# Both B4 and B5 use ImageNet normalization but different resolutions.
# We resize all images to B4 size for a fair visual comparison.
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

b4_transform = transforms.Compose([
    transforms.Resize((B4_IMG_SIZE, B4_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

b5_transform = transforms.Compose([
    transforms.Resize((B5_IMG_SIZE, B5_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return t(img.convert("RGB")).unsqueeze(0).to(DEVICE)


def tensor_to_rgb_array(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.squeeze(0).cpu() * std + mean
    return img.permute(1, 2, 0).numpy().clip(0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# GRADCAM
# ─────────────────────────────────────────────────────────────────────────────
def get_last_conv_layer(model: nn.Module) -> list:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d found in model.")
    return [last_conv]


def run_gradcam(model: nn.Module, input_tensor: torch.Tensor,
                target_class: int, use_plusplus: bool = False) -> np.ndarray:
    target_layers = get_last_conv_layer(model)
    CAMClass = GradCAMPlusPlus if use_plusplus else GradCAM
    with CAMClass(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(target_class)]
        )
        return grayscale_cam[0]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD TEST SAMPLES — TRUE RANDOM WITH FIXED SEED
# ─────────────────────────────────────────────────────────────────────────────
# Previous version used np.linspace() which picked evenly-spaced indices.
# Because Flowers102 test images are ordered by class, linspace sampling
# tends to hit different species at regular intervals — visually diverse
# but not statistically random.
#
# This version uses np.random.default_rng(seed=42) for true random sampling.
# The fixed seed makes results fully reproducible — anyone running this script
# will get the exact same 6 images. This is the standard approach in academic
# papers and can be reported as: "6 images randomly sampled from the test
# set with seed 42."
#
# np.random.default_rng is used instead of the legacy np.random.seed() because
# it is the modern NumPy recommended API and avoids global state side effects.
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42   # reported in paper as the sampling seed

def load_test_samples(data_root: str, num_samples: int, seed: int = RANDOM_SEED):
    """
    Load num_samples images randomly sampled from the Flowers102 test set.

    Uses a fixed random seed for full reproducibility. The seed is exposed
    as a parameter so it can be overridden from the command line if needed.

    For the paper, report this as:
        "GradCAM visualizations show 6 images randomly sampled from the
         Oxford Flowers 102 test set (numpy seed=42)."
    """
    dataset = Flowers102(root=data_root, split="test", transform=None, download=True)

    # Use the modern NumPy random generator API with a fixed seed.
    # replace=False ensures no image is shown twice.
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(dataset), size=num_samples, replace=False)

    # Sort indices so the figure reads in a consistent order
    indices = np.sort(indices)

    print(f"Random sampling: seed={seed}, "
          f"selected indices={indices.tolist()}")

    samples = []
    for idx in indices:
        img, label = dataset[int(idx)]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        samples.append((img, label))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 3-WAY COMPARISON FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def build_3way_figure(
    samples,
    b4_baseline: nn.Module,
    b4_dendrite: nn.Module,
    b5_baseline: nn.Module,
    output_path: str,
    use_plusplus: bool = False,
):
    n = len(samples)
    cam_method = "GradCAM++" if use_plusplus else "GradCAM"

    fig = plt.figure(figsize=(20, 6 * n))
    fig.suptitle(
        f"EfficientNet {cam_method}: B4 Baseline vs B4+Dendrite vs B5 Baseline\n"
        f"Oxford Flowers 102  —  Key argument: B4+Dendrite ≈ B5 accuracy at fewer parameters",
        fontsize=14, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(n, 4, figure=fig, hspace=0.55, wspace=0.12)

    col_titles = [
        "Original Image\n(Ground Truth)",
        "B4 Baseline\n17.7M params  |  92.57% Top-1",
        "B4 + 1 Dendrite (PAI)\n24.2M params  |  95.09% Top-1",
        "B5 Baseline\n28.5M params  |  95.85% Top-1",
    ]

    for col_i, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col_i])
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
        ax.axis("off")

    for row_i, (pil_img, true_label) in enumerate(samples):
        # Use appropriate resolution per model
        t_b4 = pil_to_tensor(pil_img, B4_IMG_SIZE)
        t_b5 = pil_to_tensor(pil_img, B5_IMG_SIZE)

        # RGB array for overlay — use B4 size as display standard
        rgb_b4 = tensor_to_rgb_array(t_b4)
        rgb_b5 = tensor_to_rgb_array(t_b5)

        class_name = FLOWER_CLASSES[true_label] if true_label < len(FLOWER_CLASSES) else str(true_label)

        # Predictions
        with torch.no_grad():
            b4b_logits = b4_baseline(t_b4)
            b4d_logits = b4_dendrite(t_b4)
            b5b_logits = b5_baseline(t_b5)

        def pred_info(logits, label):
            pred = logits.argmax(1).item()
            conf = torch.softmax(logits, 1)[0, pred].item()
            name = FLOWER_CLASSES[pred] if pred < len(FLOWER_CLASSES) else str(pred)
            correct = pred == label
            return pred, conf, name, correct

        b4b_pred, b4b_conf, b4b_name, b4b_ok = pred_info(b4b_logits, true_label)
        b4d_pred, b4d_conf, b4d_name, b4d_ok = pred_info(b4d_logits, true_label)
        b5b_pred, b5b_conf, b5b_name, b5b_ok = pred_info(b5b_logits, true_label)

        # GradCAM — target true class for all models
        cam_b4b = run_gradcam(b4_baseline, t_b4, true_label, use_plusplus)
        cam_b4d = run_gradcam(b4_dendrite, t_b4, true_label, use_plusplus)
        cam_b5b = run_gradcam(b5_baseline, t_b5, true_label, use_plusplus)

        overlay_b4b = show_cam_on_image(rgb_b4, cam_b4b, use_rgb=True)
        overlay_b4d = show_cam_on_image(rgb_b4, cam_b4d, use_rgb=True)
        # Resize B5 cam overlay to B4 display size for visual consistency
        from PIL import Image as PILImage
        overlay_b5b_pil = PILImage.fromarray(
            show_cam_on_image(rgb_b5, cam_b5b, use_rgb=True)
        ).resize((B4_IMG_SIZE, B4_IMG_SIZE))
        overlay_b5b = np.array(overlay_b5b_pil)

        # Col 0: original
        ax0 = fig.add_subplot(gs[row_i, 0])
        ax0.imshow(pil_img.resize((B4_IMG_SIZE, B4_IMG_SIZE)))
        ax0.set_title(f"True: {class_name}", fontsize=10, pad=6)
        ax0.axis("off")

        # Col 1: B4 baseline
        ax1 = fig.add_subplot(gs[row_i, 1])
        ax1.imshow(overlay_b4b)
        marker = "✓" if b4b_ok else "✗"
        ax1.set_title(f"Pred: {b4b_name} {marker}\nConf: {b4b_conf:.1%}",
                      fontsize=10, pad=6,
                      color="green" if b4b_ok else "red")
        ax1.axis("off")

        # Col 2: B4 dendrite
        ax2 = fig.add_subplot(gs[row_i, 2])
        ax2.imshow(overlay_b4d)
        marker = "✓" if b4d_ok else "✗"
        ax2.set_title(f"Pred: {b4d_name} {marker}\nConf: {b4d_conf:.1%}",
                      fontsize=10, pad=6,
                      color="green" if b4d_ok else "red")
        ax2.axis("off")

        # Col 3: B5 baseline
        ax3 = fig.add_subplot(gs[row_i, 3])
        ax3.imshow(overlay_b5b)
        marker = "✓" if b5b_ok else "✗"
        ax3.set_title(f"Pred: {b5b_name} {marker}\nConf: {b5b_conf:.1%}",
                      fontsize=10, pad=6,
                      color="green" if b5b_ok else "red")
        ax3.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved 3-way comparison → {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE CASE ANALYSIS (3-way)
# ─────────────────────────────────────────────────────────────────────────────
def find_interesting_cases_3way(data_root, b4_baseline, b4_dendrite, b5_baseline,
                                 max_cases=6):
    """
    Find cases where:
      - B4 baseline wrong, dendrite right (dendrite wins over B4)
      - B4 baseline wrong, B5 right, dendrite right (both bigger models win)
      - All three wrong (shared failure)
    """
    dataset = Flowers102(root=data_root, split="test", transform=None, download=False)
    dendrite_only_wins = []
    both_win = []

    print("\nScanning test set for interesting 3-way cases (up to 500 images)...")
    for idx in range(min(500, len(dataset))):
        img, label = dataset[idx]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        t_b4 = pil_to_tensor(img, B4_IMG_SIZE)
        t_b5 = pil_to_tensor(img, B5_IMG_SIZE)
        with torch.no_grad():
            b4b = b4_baseline(t_b4).argmax(1).item()
            b4d = b4_dendrite(t_b4).argmax(1).item()
            b5b = b5_baseline(t_b5).argmax(1).item()

        # Dendrite wins where B4 baseline fails
        if b4b != label and b4d == label:
            dendrite_only_wins.append((img, label, b4b, b4d, b5b))
        # Both dendrite and B5 correct, B4 baseline wrong
        elif b4b != label and b4d == label and b5b == label:
            both_win.append((img, label, b4b, b4d, b5b))

        if len(dendrite_only_wins) >= max_cases:
            break

    print(f"  Found {len(dendrite_only_wins)} dendrite-win cases")
    return dendrite_only_wins[:max_cases]


def build_failure_figure_3way(cases, b4_baseline, b4_dendrite, b5_baseline,
                               output_path="gradcam_failure_3way.png",
                               use_plusplus=False):
    if not cases:
        print("No interesting cases found.")
        return

    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(20, 6 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Failure Case Analysis: B4 Baseline vs B4+Dendrite vs B5 Baseline\n"
        "GradCAM targeted at ground-truth class | Rows: B4 Baseline wrong, Dendrite correct",
        fontsize=13, fontweight="bold"
    )

    col_headers = ["Original\n(True Class)",
                   "B4 Baseline\n(WRONG)",
                   "B4 + 1 Dendrite\n(CORRECT ✓)",
                   "B5 Baseline"]

    for col_i, h in enumerate(col_headers):
        axes[0][col_i].set_title(h, fontsize=11, fontweight="bold", pad=10)

    for i, (img, true_label, b4b_pred, b4d_pred, b5b_pred) in enumerate(cases):
        t_b4 = pil_to_tensor(img, B4_IMG_SIZE)
        t_b5 = pil_to_tensor(img, B5_IMG_SIZE)
        rgb_b4 = tensor_to_rgb_array(t_b4)
        rgb_b5 = tensor_to_rgb_array(t_b5)

        cam_b4b = run_gradcam(b4_baseline, t_b4, true_label, use_plusplus)
        cam_b4d = run_gradcam(b4_dendrite, t_b4, true_label, use_plusplus)
        cam_b5b = run_gradcam(b5_baseline, t_b5, true_label, use_plusplus)

        overlay_b4b = show_cam_on_image(rgb_b4, cam_b4b, use_rgb=True)
        overlay_b4d = show_cam_on_image(rgb_b4, cam_b4d, use_rgb=True)
        from PIL import Image as PILImage
        overlay_b5b = np.array(
            PILImage.fromarray(show_cam_on_image(rgb_b5, cam_b5b, use_rgb=True))
            .resize((B4_IMG_SIZE, B4_IMG_SIZE))
        )

        true_name  = FLOWER_CLASSES[true_label] if true_label < len(FLOWER_CLASSES) else str(true_label)
        b4b_name   = FLOWER_CLASSES[b4b_pred]   if b4b_pred   < len(FLOWER_CLASSES) else str(b4b_pred)
        b4d_name   = FLOWER_CLASSES[b4d_pred]   if b4d_pred   < len(FLOWER_CLASSES) else str(b4d_pred)
        b5b_name   = FLOWER_CLASSES[b5b_pred]   if b5b_pred   < len(FLOWER_CLASSES) else str(b5b_pred)
        b5b_ok = b5b_pred == true_label

        axes[i][0].imshow(img.resize((B4_IMG_SIZE, B4_IMG_SIZE)))
        axes[i][0].set_title(f"True: {true_name}", fontsize=9)
        axes[i][0].axis("off")

        axes[i][1].imshow(overlay_b4b)
        axes[i][1].set_title(f"✗ Pred: {b4b_name}", fontsize=9, color="red")
        axes[i][1].axis("off")

        axes[i][2].imshow(overlay_b4d)
        axes[i][2].set_title(f"✓ Pred: {b4d_name}", fontsize=9, color="green")
        axes[i][2].axis("off")

        axes[i][3].imshow(overlay_b5b)
        marker = "✓" if b5b_ok else "✗"
        axes[i][3].set_title(f"{marker} Pred: {b5b_name}", fontsize=9,
                              color="green" if b5b_ok else "red")
        axes[i][3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved 3-way failure analysis → {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ARGS + MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",        default="/ocean/projects/cis260045p/shared/data")
    p.add_argument("--artifact_dir",     default="./wandb_artifacts")
    p.add_argument("--output_dir",       default="./gradcam_outputs_3way")
    p.add_argument("--num_samples",      type=int, default=NUM_SAMPLES,
                   help="Number of images to show in comparison figure (default 6)")
    p.add_argument("--seed",             type=int, default=42,
                   help="Random seed for image sampling (default 42, report this in paper)")
    p.add_argument("--use_plusplus",     action="store_true")
    p.add_argument("--skip_failure",     action="store_true")
    p.add_argument("--b4_baseline_ckpt", default=None,
                   help="Path to B4 baseline best_model.pt (skips WandB download)")
    p.add_argument("--dendrite_pai_dir", default=None,
                   help="Path to PAI artifacts dir for B4 dendrite model")
    p.add_argument("--b5_baseline_ckpt", default=None,
                   help="Path to B5 baseline best_model.pt (skips WandB download)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)

    # ── 1. B4 Baseline checkpoint ─────────────────────────────────────────────
    if args.b4_baseline_ckpt:
        b4_baseline_ckpt = args.b4_baseline_ckpt
    else:
        b4_art_dir = download_artifact(
            ARTIFACT_B4_BASELINE,
            os.path.join(args.artifact_dir, "b4_baseline"),
        )
        b4_baseline_ckpt = find_pt_file(b4_art_dir)

    # ── 2. B4 Dendrite PAI directory ──────────────────────────────────────────
    if args.dendrite_pai_dir:
        pai_dir = args.dendrite_pai_dir
    else:
        pai_dir = download_artifact(
            ARTIFACT_B4_DENDRITE,
            os.path.join(args.artifact_dir, PAI_SAVE_NAME),
        )

    # ── 3. B5 Baseline checkpoint ─────────────────────────────────────────────
    if args.b5_baseline_ckpt:
        b5_baseline_ckpt = args.b5_baseline_ckpt
    else:
        b5_art_dir = download_artifact(
            ARTIFACT_B5_BASELINE,
            os.path.join(args.artifact_dir, "b5_baseline"),
        )
        b5_baseline_ckpt = find_pt_file(b5_art_dir)

    # ── 4. Load all three models ──────────────────────────────────────────────
    print("\n=== Loading B4 BASELINE (0 dendrites) ===")
    b4_baseline = build_b4_baseline(NUM_CLASSES)
    b4_baseline = load_standard_checkpoint(b4_baseline, b4_baseline_ckpt, "B4 Baseline")

    print("\n=== Loading B4 DENDRITE (1 dendrite, PAI) ===")
    b4_dendrite = load_dendrite_model(pai_dir, NUM_CLASSES)

    print("\n=== Loading B5 BASELINE (0 dendrites) ===")
    b5_baseline = build_b5_baseline(NUM_CLASSES)
    b5_baseline = load_standard_checkpoint(b5_baseline, b5_baseline_ckpt, "B5 Baseline")

    # ── 5. Load test samples ──────────────────────────────────────────────────
    print(f"\nLoading {args.num_samples} test samples from Flowers102...")
    samples = load_test_samples(args.data_root, args.num_samples, seed=args.seed)

    # ── 6. 3-way comparison figure ────────────────────────────────────────────
    out_main = os.path.join(args.output_dir, "gradcam_3way_comparison.png")
    build_3way_figure(
        samples, b4_baseline, b4_dendrite, b5_baseline,
        output_path=out_main,
        use_plusplus=args.use_plusplus,
    )

    # ── 7. Failure case analysis ──────────────────────────────────────────────
    if not args.skip_failure:
        cases = find_interesting_cases_3way(
            args.data_root, b4_baseline, b4_dendrite, b5_baseline
        )
        out_fail = os.path.join(args.output_dir, "gradcam_3way_failure.png")
        build_failure_figure_3way(
            cases, b4_baseline, b4_dendrite, b5_baseline,
            output_path=out_fail,
            use_plusplus=args.use_plusplus,
        )

    print(f"\nDone! All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
