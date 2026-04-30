"""
balanced_analysis_gradcam_3way.py
==================================
Balanced statistical analysis + GradCAM visualization comparing:
  - EfficientNet-B4 Baseline   (17.7M params | 92.57% Top-1)
  - EfficientNet-B4 + Dendrite (24.2M params | 95.09% Top-1)
  - EfficientNet-B5 Baseline   (28.5M params | 95.85% Top-1)

This script is divided into 8 clearly labeled sections:

  SECTION 1: Imports and configuration
  SECTION 2: WandB artifact downloading
  SECTION 3: Model builders (B4 baseline, B4 dendrite, B5 baseline)
  SECTION 4: Checkpoint loaders
  SECTION 5: Statistical analysis (McNemar, Kappa, error correlation)
  SECTION 6: GradCAM utilities
  SECTION 7: Balanced figure builders (4 case types + per-class accuracy)
  SECTION 8: Main entry point

Academic references:
  - Dietterich (1998). Approximate Statistical Tests for Comparing
    Supervised Classification Learning Algorithms. Neural Computation.
  - Cohen (1960). A coefficient of agreement for nominal scales.
    Educational and Psychological Measurement.
  - Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization. ICCV.

Usage:
    # First run (downloads B5 from WandB, reuses cached B4 artifacts):
    python balanced_analysis_gradcam_3way.py \\
        --data_root /ocean/projects/cis260045p/shared/data \\
        --output_dir ./gradcam_outputs_balanced \\
        --b4_baseline_ckpt ./wandb_artifacts/baseline/best_model.pt \\
        --dendrite_pai_dir ./wandb_artifacts/artifacts_efficientnet_b4_flowers102 \\
        --b5_baseline_ckpt ./wandb_artifacts/b5_baseline/best_model.pt

    # To also run GradCAM visualizations (slower, requires GPU):
    python balanced_analysis_gradcam_3way.py \\
        --data_root /ocean/projects/cis260045p/shared/data \\
        --output_dir ./gradcam_outputs_balanced \\
        --b4_baseline_ckpt ./wandb_artifacts/baseline/best_model.pt \\
        --dendrite_pai_dir ./wandb_artifacts/artifacts_efficientnet_b4_flowers102 \\
        --b5_baseline_ckpt ./wandb_artifacts/b5_baseline/best_model.pt \\
        --run_gradcam
"""

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS AND CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# We import standard ML libraries plus scipy for statistical tests.
# statsmodels provides McNemar's test. sklearn provides Cohen's Kappa.
# seaborn is used for the per-class accuracy heatmap.
# ═════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as tv
from torchvision.datasets import Flowers102

import wandb

# Statistical testing libraries
from scipy.stats import chi2                     # for McNemar's exact test fallback
from statsmodels.stats.contingency_tables import mcnemar   # McNemar's test
from sklearn.metrics import cohen_kappa_score, confusion_matrix  # Kappa + confusion

# GradCAM
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# PerforatedAI — needed to reconstruct the dendrite model
try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    PAI_AVAILABLE = True
except ImportError:
    PAI_AVAILABLE = False
    print("[WARNING] PerforatedAI not found. Dendrite model cannot be loaded.")

# ── Constants ─────────────────────────────────────────────────────────────────
# These match the exact WandB artifact paths and training configurations.

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

NUM_CLASSES  = 102
B4_IMG_SIZE  = 380   # EfficientNet-B4 native resolution
B5_IMG_SIZE  = 456   # EfficientNet-B5 native resolution
NUM_SAMPLES  = 10    # GradCAM figure rows (10 randomly sampled with seed=42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# All 102 Oxford Flowers class names in label order
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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: WANDB ARTIFACT DOWNLOADING
# ─────────────────────────────────────────────────────────────────────────────
# These functions handle downloading model checkpoints from WandB when local
# copies are not available. The find_pt_file helper locates the .pt file
# inside whatever subdirectory WandB creates during download.
# ═════════════════════════════════════════════════════════════════════════════

def download_artifact(artifact_path: str, download_dir: str) -> str:
    """Download a WandB artifact and return the local directory path."""
    print(f"\nDownloading WandB artifact: {artifact_path}")
    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")
    local_dir = artifact.download(root=download_dir)
    print(f"  Saved to: {local_dir}")
    return local_dir


def find_pt_file(directory: str) -> str:
    """Recursively find the first .pt or .pth file in a directory."""
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".pt") or f.endswith(".pth"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No .pt file found in {directory}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
# Each model was saved with a specific architecture. We must reconstruct that
# exact architecture before loading weights, otherwise key names won't match.
#
# B4 Baseline: standard torchvision EfficientNet-B4 with classifier replaced
# B4 Dendrite: has an extra pre_fc hidden layer (1792→1792) added by PAI's
#              wrap_model() before the final 102-class output layer
# B5 Baseline: standard torchvision EfficientNet-B5 with classifier replaced
# ═════════════════════════════════════════════════════════════════════════════

def build_b4_baseline(num_classes: int = 102) -> nn.Module:
    """
    Standard EfficientNet-B4 for the baseline run.
    Checkpoint keys: features.*.weight, classifier.1.weight/bias
    """
    model = tv.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_b4_with_pre_fc(num_classes: int = 102) -> nn.Module:
    """
    EfficientNet-B4 with PAI's pre_fc hidden layer.
    During training, wrap_model() inserted a 1792→1792 linear layer
    (called pre_fc) between the backbone and the final classifier.
    Checkpoint keys: pre_fc.* (1792→1792) and classifier_fc.* (1792→102)
    The forward function is overridden to route through both layers.
    """
    model = tv.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features  # 1792 for B4

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, in_features),  # pre_fc: 1792→1792
    )
    model.classifier_fc = nn.Linear(in_features, num_classes)  # 1792→102

    def new_forward(x):
        x = model.features(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.classifier(x)      # dropout + pre_fc
        x = model.classifier_fc(x)   # final class scores
        return x

    model.forward = new_forward
    return model


def build_b5_baseline(num_classes: int = 102) -> nn.Module:
    """
    Standard EfficientNet-B5 for the B5 baseline run.
    Checkpoint keys: features.*.weight, classifier.1.weight/bias
    """
    model = tv.efficientnet_b5(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: CHECKPOINT LOADERS
# ─────────────────────────────────────────────────────────────────────────────
# Two different loading strategies are needed:
#
# load_standard_checkpoint: for B4 and B5 baselines, which were saved as
#   standard PyTorch dicts with a "model_state_dict" key.
#
# load_dendrite_model: for the PAI dendrite model, which was saved in
#   safetensors format with PAI-specific key names (pre_fc.*, classifier_fc.*).
#   We must also re-initialize PAI's scaffolding before loading weights,
#   matching the exact GPA settings used during training.
# ═════════════════════════════════════════════════════════════════════════════

def load_standard_checkpoint(model: nn.Module, checkpoint_path: str,
                              label: str = "model") -> nn.Module:
    """
    Load a standard PyTorch checkpoint.
    Handles: {"model_state_dict": ...} format from main_baseline.py
    Also strips DataParallel "module." prefix if present.
    """
    print(f"\nLoading {label} checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Remove DataParallel/DDP "module." prefix if checkpoint was saved
    # from a multi-GPU training run
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if "num_batches" not in k]
    if real_missing:
        print(f"  [WARN] Missing keys ({len(real_missing)}): {real_missing[:3]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    model.eval()
    model.to(DEVICE)
    print(f"  {label} loaded successfully.")
    return model


def load_dendrite_model(pai_dir: str, num_classes: int = 102) -> nn.Module:
    """
    Load the PAI-trained B4 dendrite model.

    Why this is complex: PAI saves the model in safetensors format with its
    own internal key naming (pre_fc.* instead of classifier.1.*). We must:
      1. Build the correct pre_fc architecture
      2. Re-initialize PAI's GPA scaffolding with the same settings as training
      3. Load weights from safetensors with manual key remapping
      4. Skip the tracker_string buffer (PAI metadata, not model weights)

    The GPA settings here must exactly match the training run config:
      --dendrite-mode 1  --max-dendrites 1  --improvement-threshold 1
      --pai-forward-function relu  --candidate-weight-init-mult 0.1
      --epochs 50  (switch interval = 50 // (1+1) = 25)
    """
    if not PAI_AVAILABLE:
        raise RuntimeError(
            "PerforatedAI is required. "
            "Run: conda activate /ocean/projects/cis260045p/shared/perf_ai"
        )

    from safetensors import safe_open
    import torch.nn.functional as F

    print(f"\nBuilding B4 dendrite model from: {pai_dir}")

    # Build the pre_fc architecture that matches training
    base_model = build_b4_with_pre_fc(num_classes)

    # Re-initialize PAI's global config to match training settings.
    # GPA.pc is PAI's global config object. These settings determine how
    # PAI wraps the model layers — they must match or the key structure
    # won't align with what's in the checkpoint.
    GPA.pc.set_verbose(False)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_module_names_to_perforate([])

    # The dendrite was placed on classifier.1 (the pre_fc layer)
    convert_module_id = ".classifier.1"
    GPA.pc.set_module_ids_to_perforate([convert_module_id])

    # Track all leaf modules except the one being perforated
    ids_to_track = [
        f".{name}" for name, module in base_model.named_modules()
        if len(list(module.children())) == 0
        and name != convert_module_id.lstrip(".")
    ]
    GPA.pc.set_module_ids_to_track(ids_to_track)

    if hasattr(GPA.pc, "set_weight_decay_accepted"):
        GPA.pc.set_weight_decay_accepted(True)

    # improvement_threshold=1 maps to preset [0.001, 0.0001, 0]
    GPA.pc.set_improvement_threshold([0.001, 0.0001, 0])
    GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
    GPA.pc.set_fixed_switch_num(25)       # epochs // (max_dendrites + 1)
    GPA.pc.set_first_fixed_switch_num(25)
    GPA.pc.set_candidate_weight_initialization_multiplier(0.1)
    GPA.pc.set_max_dendrites(1)
    GPA.pc.set_pai_forward_function(F.relu)
    if hasattr(GPA.pc, "set_perforated_backpropagation"):
        GPA.pc.set_perforated_backpropagation(False)  # dendrite_mode=1, not 2

    # initialize_pai wraps the model with PAI's dendrite scaffolding
    print("  Initializing PAI scaffold...")
    model = UPA.initialize_pai(
        base_model, doing_pai=True, save_name=pai_dir,
        making_graphs=False, maximizing_score=True, num_classes=num_classes,
    )

    # Load weights from safetensors, remapping PAI's internal key names
    # to match our initialized model's key structure
    print("  Loading weights from safetensors checkpoint...")
    ckpt_path = os.path.join(pai_dir, "best_model.pt")
    tensors = {}
    with safe_open(ckpt_path, framework="pt", device=str(DEVICE)) as f:
        for key in f.keys():
            # Skip PAI's internal tracking buffer — not a model weight
            if key == "tracker_string":
                continue
            # Remap PAI's internal name to our model's name
            new_key = key.replace("pre_fc.", "classifier.1.")
            tensors[new_key] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(tensors, strict=False)
    # Filter out non-critical PAI tracking tensors from the warning
    real_missing = [k for k in missing if not any(x in k for x in
                    ["num_batches", "tracker", "this_node",
                     "this_output", "dendrites_to_top"])]
    real_unexpected = [k for k in unexpected if not any(x in k for x in
                       ["tracker_string", "this_node", "this_output",
                        "dendrites_to_top"])]
    if real_missing:
        print(f"  [WARN] Missing ({len(real_missing)}): {real_missing[:3]}")
    if real_unexpected:
        print(f"  [WARN] Unexpected ({len(real_unexpected)}): {real_unexpected[:3]}")

    model.eval()
    model.to(DEVICE)
    print("  B4 dendrite model loaded successfully.")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
# This is the core scientific contribution of this script. We run four
# complementary analyses on the FULL test set predictions:
#
# 5a. run_full_test_set_predictions()
#     Runs all three models on every test image and collects predictions.
#     This is the ground truth for all statistical tests.
#
# 5b. mcnemar_test()
#     Tests whether the PATTERN of errors differs significantly between
#     two models, not just the accuracy numbers. Uses a 2x2 contingency
#     table of (model A correct, model B correct) combinations.
#     Reference: Dietterich (1998)
#
# 5c. cohen_kappa()
#     Measures agreement between predictions and ground truth, corrected
#     for the probability of chance agreement. More robust than accuracy
#     for imbalanced datasets like Flowers102.
#     Reference: Cohen (1960)
#
# 5d. error_correlation()
#     Computes how correlated the error patterns are between model pairs.
#     High correlation = models make same mistakes = similar representations.
#     Low correlation = models learned complementary features.
#
# 5e. per_class_accuracy()
#     Computes per-class accuracy for all 102 flower categories.
#     Reveals systematic weaknesses invisible in aggregate accuracy.
#
# 5f. generate_statistical_report()
#     Combines all analyses into a printed report + JSON file.
# ═════════════════════════════════════════════════════════════════════════════

def pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """Convert PIL image to normalized tensor at given resolution."""
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return t(img.convert("RGB")).unsqueeze(0).to(DEVICE)


def tensor_to_rgb_array(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize tensor back to [0,1] float32 RGB array for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.squeeze(0).cpu() * std + mean
    return img.permute(1, 2, 0).numpy().clip(0, 1).astype(np.float32)


def run_full_test_set_predictions(data_root, b4_baseline, b4_dendrite, b5_baseline,
                                   max_images=None):
    """
    Section 5a: Run all three models on the entire test set.

    Returns arrays of shape (N,) for:
      - true_labels: ground truth class indices
      - b4b_preds, b4d_preds, b5b_preds: predicted class indices
      - b4b_correct, b4d_correct, b5b_correct: boolean arrays

    We use max_images to optionally limit evaluation for speed during testing.
    For final paper results, leave max_images=None to use all 6149 test images.
    """
    dataset = Flowers102(root=data_root, split="test", transform=None, download=False)
    n = len(dataset) if max_images is None else min(max_images, len(dataset))

    true_labels = np.zeros(n, dtype=int)
    b4b_preds   = np.zeros(n, dtype=int)
    b4d_preds   = np.zeros(n, dtype=int)
    b5b_preds   = np.zeros(n, dtype=int)

    print(f"\nRunning full test set evaluation ({n} images)...")
    print("This may take 5-10 minutes on GPU. Progress shown every 500 images.")

    for idx in range(n):
        if idx % 500 == 0:
            print(f"  [{idx}/{n}] evaluating...")

        img, label = dataset[idx]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        # B4 models use 380px, B5 uses 456px (their native training resolutions)
        t_b4 = pil_to_tensor(img, B4_IMG_SIZE)
        t_b5 = pil_to_tensor(img, B5_IMG_SIZE)

        with torch.no_grad():
            b4b_preds[idx] = b4_baseline(t_b4).argmax(1).item()
            b4d_preds[idx] = b4_dendrite(t_b4).argmax(1).item()
            b5b_preds[idx] = b5_baseline(t_b5).argmax(1).item()

        true_labels[idx] = label

    b4b_correct = (b4b_preds == true_labels)
    b4d_correct = (b4d_preds == true_labels)
    b5b_correct = (b5b_preds == true_labels)

    print(f"\n  B4 Baseline accuracy:  {b4b_correct.mean():.4f} "
          f"({b4b_correct.sum()}/{n})")
    print(f"  B4 Dendrite accuracy:  {b4d_correct.mean():.4f} "
          f"({b4d_correct.sum()}/{n})")
    print(f"  B5 Baseline accuracy:  {b5b_correct.mean():.4f} "
          f"({b5b_correct.sum()}/{n})")

    return true_labels, b4b_preds, b4d_preds, b5b_preds, \
           b4b_correct, b4d_correct, b5b_correct


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray,
                 label_a: str, label_b: str) -> dict:
    """
    Section 5b: McNemar's test for comparing two classifiers.

    McNemar's test asks: "Are the differences in errors between model A
    and model B statistically significant, or could they be due to chance?"

    It builds a 2x2 contingency table:
        b = cases where A correct, B wrong  (A wins)
        c = cases where A wrong, B correct  (B wins)

    The test statistic focuses only on the DISCORDANT cases (b and c),
    because concordant cases (both right or both wrong) carry no information
    about which model is better.

    H0: The models have the same error rate (b == c in the population)
    H1: The models have different error rates

    If p < 0.05, the improvement is statistically significant.

    Reference: Dietterich (1998). Neural Computation 10(7):1895-1923.
    """
    # Build contingency table
    # [both_wrong, a_wrong_b_right]
    # [a_right_b_wrong, both_right]
    both_correct   = np.sum(correct_a & correct_b)
    a_only_correct = np.sum(correct_a & ~correct_b)   # b: A wins
    b_only_correct = np.sum(~correct_a & correct_b)   # c: B wins
    both_wrong     = np.sum(~correct_a & ~correct_b)

    table = np.array([
        [both_wrong,     b_only_correct],
        [a_only_correct, both_correct  ],
    ])

    # Run McNemar's test with continuity correction
    # (Edwards correction recommended when b+c < 25)
    result = mcnemar(table, exact=False, correction=True)
    p_value = result.pvalue
    statistic = result.statistic

    significant = p_value < 0.05

    print(f"\n  McNemar's Test: {label_a} vs {label_b}")
    print(f"    Contingency table:")
    print(f"      Both correct:      {both_correct}")
    print(f"      {label_a} only correct: {a_only_correct}")
    print(f"      {label_b} only correct: {b_only_correct}")
    print(f"      Both wrong:        {both_wrong}")
    print(f"    χ² statistic: {statistic:.4f}")
    print(f"    p-value:      {p_value:.6f}  "
          f"({'SIGNIFICANT ✓' if significant else 'not significant ✗'})")

    return {
        "comparison": f"{label_a} vs {label_b}",
        "both_correct": int(both_correct),
        "a_only_correct": int(a_only_correct),
        "b_only_correct": int(b_only_correct),
        "both_wrong": int(both_wrong),
        "chi2_statistic": float(statistic),
        "p_value": float(p_value),
        "significant_p05": bool(significant),
    }


def cohen_kappa(true_labels: np.ndarray, preds: np.ndarray,
                label: str) -> dict:
    """
    Section 5c: Cohen's Kappa coefficient.

    Kappa measures how much better than chance a classifier performs.
    Unlike accuracy, it accounts for the possibility that some correct
    predictions happen by random chance given the class distribution.

    For Flowers102 (102 classes, roughly balanced), accuracy and kappa
    should be similar, but kappa is more defensible in academic papers
    because it's explicitly chance-corrected.

    Interpretation:
        κ < 0.20: slight agreement
        κ = 0.21–0.40: fair agreement
        κ = 0.41–0.60: moderate agreement
        κ = 0.61–0.80: substantial agreement
        κ > 0.80: almost perfect agreement

    Reference: Cohen (1960). Educational and Psychological Measurement 20(1).
    """
    kappa = cohen_kappa_score(true_labels, preds)
    accuracy = np.mean(true_labels == preds)

    print(f"\n  Cohen's Kappa: {label}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Kappa:    {kappa:.4f}")

    return {
        "model": label,
        "accuracy": float(accuracy),
        "kappa": float(kappa),
    }


def error_correlation(correct_a: np.ndarray, correct_b: np.ndarray,
                      label_a: str, label_b: str) -> dict:
    """
    Section 5d: Error correlation between model pairs.

    This converts each model's correctness array (True/False per image)
    into an error array (1=wrong, 0=right), then computes the Pearson
    correlation coefficient between the two error arrays.

    High correlation (close to 1.0):
        The models make mistakes on the SAME images.
        This suggests they learned similar representations.
        The dendrite didn't fundamentally change what the model attends to.

    Low correlation (close to 0.0):
        The models make mistakes on DIFFERENT images.
        This suggests complementary representations were learned.
        The dendrite genuinely changed the model's feature utilization.

    For your paper argument: a low B4-vs-Dendrite correlation AND
    a higher Dendrite-vs-B5 correlation would strongly support the claim
    that the dendrite makes B4 more like B5 in terms of what it learns.
    """
    errors_a = (~correct_a).astype(float)
    errors_b = (~correct_b).astype(float)

    # Pearson correlation
    corr = float(np.corrcoef(errors_a, errors_b)[0, 1])

    # Also compute how many errors are shared vs unique
    shared_errors  = np.sum(~correct_a & ~correct_b)
    only_a_errors  = np.sum(~correct_a & correct_b)
    only_b_errors  = np.sum(correct_a & ~correct_b)

    print(f"\n  Error Correlation: {label_a} vs {label_b}")
    print(f"    Pearson r:         {corr:.4f}")
    print(f"    Shared errors:     {shared_errors}")
    print(f"    Only {label_a} wrong: {only_a_errors}")
    print(f"    Only {label_b} wrong: {only_b_errors}")

    return {
        "comparison": f"{label_a} vs {label_b}",
        "pearson_r": corr,
        "shared_errors": int(shared_errors),
        "only_a_errors": int(only_a_errors),
        "only_b_errors": int(only_b_errors),
    }


def per_class_accuracy(true_labels: np.ndarray, b4b_preds: np.ndarray,
                       b4d_preds: np.ndarray, b5b_preds: np.ndarray,
                       output_dir: str):
    """
    Section 5e: Per-class accuracy breakdown across all 102 flower categories.

    This reveals systematic weaknesses that aggregate accuracy hides.
    For example, a model might be 99% accurate on roses but only 50%
    accurate on visually similar species like garden phlox and carnations.

    We produce two outputs:
      1. A heatmap showing per-class accuracy for all three models side by side
      2. A bar chart of the classes where dendrite most improves over B4 baseline

    This figure is directly analogous to analyses in fine-grained recognition
    papers (e.g. He et al. on iNaturalist) where per-species accuracy is
    the primary metric.
    """
    n_classes = NUM_CLASSES

    # Compute per-class accuracy for each model
    b4b_acc = np.zeros(n_classes)
    b4d_acc = np.zeros(n_classes)
    b5b_acc = np.zeros(n_classes)
    class_counts = np.zeros(n_classes, dtype=int)

    for c in range(n_classes):
        mask = true_labels == c
        count = mask.sum()
        class_counts[c] = count
        if count > 0:
            b4b_acc[c] = (b4b_preds[mask] == c).mean()
            b4d_acc[c] = (b4d_preds[mask] == c).mean()
            b5b_acc[c] = (b5b_preds[mask] == c).mean()

    # ── Figure 1: Per-class accuracy heatmap ─────────────────────────────────
    # Sort classes by B4 baseline accuracy so patterns are visible
    sort_idx = np.argsort(b4b_acc)
    sorted_names = [FLOWER_CLASSES[i] for i in sort_idx]
    data = np.vstack([b4b_acc[sort_idx], b4d_acc[sort_idx], b5b_acc[sort_idx]])

    fig, ax = plt.subplots(figsize=(22, 10))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=6)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["B4 Baseline\n17.7M", "B4+Dendrite\n24.2M",
                         "B5 Baseline\n28.5M"], fontsize=11)
    ax.set_title(
        "Per-class Top-1 Accuracy: B4 Baseline vs B4+Dendrite vs B5 Baseline\n"
        "Oxford Flowers 102 (sorted by B4 Baseline accuracy, low → high)",
        fontsize=13, fontweight="bold", pad=12
    )
    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    out = os.path.join(output_dir, "per_class_accuracy_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved per-class heatmap → {out}")

    # ── Figure 2: Top dendrite improvement classes ────────────────────────────
    # Find the classes where the dendrite improved most over B4 baseline.
    # These are the most interesting classes for qualitative GradCAM analysis.
    dendrite_gain = b4d_acc - b4b_acc
    top_gain_idx  = np.argsort(dendrite_gain)[-20:][::-1]
    top_loss_idx  = np.argsort(dendrite_gain)[:20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Top gains
    gains = dendrite_gain[top_gain_idx]
    names = [FLOWER_CLASSES[i] for i in top_gain_idx]
    bars = ax1.barh(range(len(gains)), gains, color="#1D9E75")
    ax1.set_yticks(range(len(gains)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("Accuracy gain (Dendrite - B4 Baseline)", fontsize=11)
    ax1.set_title("Top 20 classes: dendrite improvement over B4 baseline",
                  fontsize=11, fontweight="bold")
    ax1.axvline(0, color="black", linewidth=0.5)

    # Top losses (where dendrite hurts)
    losses = dendrite_gain[top_loss_idx]
    names_l = [FLOWER_CLASSES[i] for i in top_loss_idx]
    ax2.barh(range(len(losses)), losses, color="#D85A30")
    ax2.set_yticks(range(len(losses)))
    ax2.set_yticklabels(names_l, fontsize=9)
    ax2.set_xlabel("Accuracy change (Dendrite - B4 Baseline)", fontsize=11)
    ax2.set_title("Top 20 classes: dendrite regression vs B4 baseline",
                  fontsize=11, fontweight="bold")
    ax2.axvline(0, color="black", linewidth=0.5)

    plt.suptitle(
        "Dendrite per-class accuracy gain/loss vs B4 Baseline\n"
        "Gain = dendrite helps | Loss = dendrite hurts on this class",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out2 = os.path.join(output_dir, "per_class_dendrite_gain.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved dendrite gain chart → {out2}")

    return b4b_acc, b4d_acc, b5b_acc, dendrite_gain


def generate_statistical_report(true_labels, b4b_preds, b4d_preds, b5b_preds,
                                  b4b_correct, b4d_correct, b5b_correct,
                                  output_dir: str):
    """
    Section 5f: Run all statistical tests and generate the full report.

    Produces:
      - Console output with all test results
      - statistical_report.json: machine-readable results for paper tables
      - error_overlap_venn.png: visual showing shared/unique errors
    """
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS REPORT")
    print("="*60)

    results = {}

    # McNemar's tests — all three pairwise comparisons
    print("\n── McNemar's Tests ──────────────────────────────────────")
    results["mcnemar"] = {
        "b4_vs_dendrite": mcnemar_test(b4b_correct, b4d_correct,
                                        "B4 Baseline", "B4 Dendrite"),
        "b4_vs_b5":       mcnemar_test(b4b_correct, b5b_correct,
                                        "B4 Baseline", "B5 Baseline"),
        "dendrite_vs_b5": mcnemar_test(b4d_correct, b5b_correct,
                                        "B4 Dendrite", "B5 Baseline"),
    }

    # Cohen's Kappa for all three models
    print("\n── Cohen's Kappa ─────────────────────────────────────────")
    results["kappa"] = {
        "b4_baseline": cohen_kappa(true_labels, b4b_preds, "B4 Baseline"),
        "b4_dendrite": cohen_kappa(true_labels, b4d_preds, "B4 Dendrite"),
        "b5_baseline": cohen_kappa(true_labels, b5b_preds, "B5 Baseline"),
    }

    # Error correlation
    print("\n── Error Correlation ────────────────────────────────────")
    results["error_correlation"] = {
        "b4_vs_dendrite": error_correlation(b4b_correct, b4d_correct,
                                             "B4 Baseline", "B4 Dendrite"),
        "b4_vs_b5":       error_correlation(b4b_correct, b5b_correct,
                                             "B4 Baseline", "B5 Baseline"),
        "dendrite_vs_b5": error_correlation(b4d_correct, b5b_correct,
                                             "B4 Dendrite", "B5 Baseline"),
    }

    # Overall accuracy summary
    results["accuracy"] = {
        "b4_baseline": float(b4b_correct.mean()),
        "b4_dendrite": float(b4d_correct.mean()),
        "b5_baseline": float(b5b_correct.mean()),
    }

    # Save JSON report for easy copy-paste into paper tables
    report_path = os.path.join(output_dir, "statistical_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved statistical report → {report_path}")

    # ── Error overlap Venn diagram ────────────────────────────────────────────
    # Shows how many errors are shared vs unique between all three models.
    # This is the visual companion to the error correlation numbers.
    n = len(true_labels)
    b4b_err = ~b4b_correct
    b4d_err = ~b4d_correct
    b5b_err = ~b5b_correct

    only_b4b  = np.sum(b4b_err & ~b4d_err & ~b5b_err)
    only_b4d  = np.sum(~b4b_err & b4d_err & ~b5b_err)
    only_b5b  = np.sum(~b4b_err & ~b4d_err & b5b_err)
    b4b_b4d   = np.sum(b4b_err & b4d_err & ~b5b_err)
    b4b_b5b   = np.sum(b4b_err & ~b4d_err & b5b_err)
    b4d_b5b   = np.sum(~b4b_err & b4d_err & b5b_err)
    all_three = np.sum(b4b_err & b4d_err & b5b_err)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Draw three overlapping circles
    c1 = plt.Circle((3.5, 4.5), 2.5, alpha=0.25, color="#378ADD", label="B4 Baseline")
    c2 = plt.Circle((5.0, 4.5), 2.5, alpha=0.25, color="#1D9E75", label="B4+Dendrite")
    c3 = plt.Circle((4.25, 3.0), 2.5, alpha=0.25, color="#D85A30", label="B5 Baseline")
    for c in [c1, c2, c3]:
        ax.add_patch(c)

    # Labels inside each region
    ax.text(2.2, 5.2, f"{only_b4b}", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#0C447C")
    ax.text(6.2, 5.2, f"{only_b4d}", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#085041")
    ax.text(4.25, 1.4, f"{only_b5b}", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#712B13")
    ax.text(3.8, 5.0, f"{b4b_b4d}", ha="center", va="center",
            fontsize=11, color="#333")
    ax.text(2.8, 3.5, f"{b4b_b5b}", ha="center", va="center",
            fontsize=11, color="#333")
    ax.text(5.6, 3.5, f"{b4d_b5b}", ha="center", va="center",
            fontsize=11, color="#333")
    ax.text(4.25, 4.0, f"{all_three}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333")

    # Circle labels
    ax.text(1.5, 6.8, "B4 Baseline\n17.7M params", ha="center",
            fontsize=10, color="#0C447C", fontweight="bold")
    ax.text(7.0, 6.8, "B4+Dendrite\n24.2M params", ha="center",
            fontsize=10, color="#085041", fontweight="bold")
    ax.text(4.25, 0.2, "B5 Baseline\n28.5M params", ha="center",
            fontsize=10, color="#712B13", fontweight="bold")

    ax.set_title(
        "Error overlap across all three models\n"
        "Numbers = images misclassified by that model or combination",
        fontsize=12, fontweight="bold"
    )
    out3 = os.path.join(output_dir, "error_overlap_venn.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved error overlap Venn → {out3}")

    # Print paper-ready summary
    print("\n" + "="*60)
    print("PAPER-READY SUMMARY")
    print("="*60)
    mc_b4_dend = results["mcnemar"]["b4_vs_dendrite"]
    mc_dend_b5 = results["mcnemar"]["dendrite_vs_b5"]
    kappa_b4d  = results["kappa"]["b4_dendrite"]["kappa"]
    kappa_b5b  = results["kappa"]["b5_baseline"]["kappa"]
    ec_b4_dend = results["error_correlation"]["b4_vs_dendrite"]["pearson_r"]
    ec_dend_b5 = results["error_correlation"]["dendrite_vs_b5"]["pearson_r"]

    print(f"""
McNemar's test confirms the B4+Dendrite model's improvement over B4 Baseline
is {'statistically significant' if mc_b4_dend['significant_p05'] else 'NOT statistically significant'}
(χ²={mc_b4_dend['chi2_statistic']:.3f}, p={mc_b4_dend['p_value']:.4f}).

The difference between B4+Dendrite and B5 Baseline is
{'statistically significant' if mc_dend_b5['significant_p05'] else 'NOT statistically significant'}
(χ²={mc_dend_b5['chi2_statistic']:.3f}, p={mc_dend_b5['p_value']:.4f}),
{'suggesting the dendrite model does NOT reach B5 performance.' if mc_dend_b5['significant_p05']
 else 'supporting the claim that B4+Dendrite achieves statistically equivalent performance to B5.'}

Cohen's Kappa: B4+Dendrite={kappa_b4d:.4f}, B5 Baseline={kappa_b5b:.4f}

Error correlation:
  B4 Baseline vs B4+Dendrite: r={ec_b4_dend:.4f}
  B4+Dendrite vs B5 Baseline: r={ec_dend_b5:.4f}
{'  → Dendrite errors more similar to B5 than to B4 baseline' if ec_dend_b5 > ec_b4_dend
 else '  → Dendrite errors more similar to B4 baseline than to B5'}
""")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: GRADCAM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
# Standard GradCAM helpers. get_last_conv_layer() dynamically finds the
# correct target layer rather than hardcoding a path, which is necessary
# because PAI restructures the classifier head and can shift layer names.
# ═════════════════════════════════════════════════════════════════════════════

def get_last_conv_layer(model: nn.Module) -> list:
    """
    Dynamically find the last Conv2d in the model.
    This is robust to PAI's layer restructuring — PAI only modifies the
    classifier head, so the last Conv2d is always in the backbone and
    always a valid GradCAM target regardless of PAI wrapping.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d found in model.")
    return [last_conv]


def run_gradcam(model: nn.Module, input_tensor: torch.Tensor,
                target_class: int, use_plusplus: bool = False) -> np.ndarray:
    """
    Run GradCAM on a single image tensor.
    target_class is the TRUE label, not the predicted label —
    this shows what each model uses to recognize the correct class
    regardless of whether it predicted correctly.
    """
    target_layers = get_last_conv_layer(model)
    CAMClass = GradCAMPlusPlus if use_plusplus else GradCAM
    with CAMClass(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(target_class)]
        )
        return grayscale_cam[0]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: BALANCED FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
# Four case types for a balanced qualitative analysis:
#
#   Type 1: B4 wrong, Dendrite right            → "Dendrite wins over B4"
#   Type 2: Dendrite wrong, B4 right            → "B4 wins over Dendrite"
#   Type 3: Dendrite wrong, B5 right            → "B5 wins over Dendrite"
#   Type 4: All three wrong                     → "Shared failure"
#
# For each type we show GradCAM heatmaps for all three models side by side,
# targeted at the ground truth class. This makes it easy to see whether
# attention quality correlates with prediction correctness.
#
# We also build the standard 3-way random sample comparison figure.
# ═════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42   # fixed seed for reproducibility

def load_test_samples(data_root: str, num_samples: int, seed: int = RANDOM_SEED):
    """
    Load num_samples images randomly sampled with fixed seed=42.
    Reproducible and defensible for academic papers.
    Report as: 10 images randomly sampled using numpy.random.default_rng(seed=42).
    """
    dataset = Flowers102(root=data_root, split="test", transform=None, download=True)
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(dataset), size=num_samples, replace=False)
    indices = np.sort(indices)
    print(f"  Random sampling: seed={seed}, selected indices={indices.tolist()}")
    samples = []
    for idx in indices:
        img, label = dataset[int(idx)]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        samples.append((img, label))
    return samples


def find_balanced_cases(data_root, b4_baseline, b4_dendrite, b5_baseline,
                         cases_per_type=3, max_scan=1000):
    """
    Scan the test set to find balanced examples of all four case types.

    cases_per_type: how many examples of each type to collect (default 3)
    max_scan: maximum images to scan before stopping

    Returns a dict with keys: dendrite_wins, b4_wins, b5_wins, all_wrong
    Each value is a list of (img, true_label, b4b_pred, b4d_pred, b5b_pred)
    """
    dataset = Flowers102(root=data_root, split="test", transform=None, download=False)
    cases = {
        "dendrite_wins": [],  # B4 wrong, dendrite right
        "b4_wins":       [],  # Dendrite wrong, B4 right
        "b5_wins":       [],  # Dendrite wrong, B5 right
        "all_wrong":     [],  # All three wrong
    }

    print(f"\nScanning for balanced cases (up to {max_scan} images)...")
    for idx in range(min(max_scan, len(dataset))):
        img, label = dataset[idx]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        t_b4 = pil_to_tensor(img, B4_IMG_SIZE)
        t_b5 = pil_to_tensor(img, B5_IMG_SIZE)
        with torch.no_grad():
            b4b = b4_baseline(t_b4).argmax(1).item()
            b4d = b4_dendrite(t_b4).argmax(1).item()
            b5b = b5_baseline(t_b5).argmax(1).item()

        entry = (img, label, b4b, b4d, b5b)

        if b4b != label and b4d == label:
            cases["dendrite_wins"].append(entry)
        elif b4b == label and b4d != label:
            cases["b4_wins"].append(entry)
        elif b4d != label and b5b == label:
            cases["b5_wins"].append(entry)
        elif b4b != label and b4d != label and b5b != label:
            cases["all_wrong"].append(entry)

        # Stop once we have enough of every type
        if all(len(v) >= cases_per_type for v in cases.values()):
            break

    for k, v in cases.items():
        print(f"  {k}: {len(v)} cases found")

    # Trim to cases_per_type
    return {k: v[:cases_per_type] for k, v in cases.items()}


def build_gradcam_row(ax_row, img, true_label, b4b_pred, b4d_pred, b5b_pred,
                       b4_baseline, b4_dendrite, b5_baseline,
                       case_type_label, use_plusplus=False):
    """
    Helper: fill one row of the balanced figure with 4 panels:
    [original] [B4 baseline GradCAM] [B4 dendrite GradCAM] [B5 baseline GradCAM]
    """
    t_b4   = pil_to_tensor(img, B4_IMG_SIZE)
    t_b5   = pil_to_tensor(img, B5_IMG_SIZE)
    rgb_b4 = tensor_to_rgb_array(t_b4)
    rgb_b5 = tensor_to_rgb_array(t_b5)

    cam_b4b = run_gradcam(b4_baseline, t_b4, true_label, use_plusplus)
    cam_b4d = run_gradcam(b4_dendrite, t_b4, true_label, use_plusplus)
    cam_b5b = run_gradcam(b5_baseline, t_b5, true_label, use_plusplus)

    ov_b4b = show_cam_on_image(rgb_b4, cam_b4b, use_rgb=True)
    ov_b4d = show_cam_on_image(rgb_b4, cam_b4d, use_rgb=True)
    # Resize B5 overlay to B4 display size for visual consistency
    ov_b5b = np.array(
        Image.fromarray(show_cam_on_image(rgb_b5, cam_b5b, use_rgb=True))
        .resize((B4_IMG_SIZE, B4_IMG_SIZE))
    )

    true_name = FLOWER_CLASSES[true_label] if true_label < len(FLOWER_CLASSES) else str(true_label)
    b4b_name  = FLOWER_CLASSES[b4b_pred]   if b4b_pred   < len(FLOWER_CLASSES) else str(b4b_pred)
    b4d_name  = FLOWER_CLASSES[b4d_pred]   if b4d_pred   < len(FLOWER_CLASSES) else str(b4d_pred)
    b5b_name  = FLOWER_CLASSES[b5b_pred]   if b5b_pred   < len(FLOWER_CLASSES) else str(b5b_pred)

    # Col 0: original image with case type label
    ax_row[0].imshow(img.resize((B4_IMG_SIZE, B4_IMG_SIZE)))
    ax_row[0].set_title(f"[{case_type_label}]\nTrue: {true_name}", fontsize=8, pad=4)
    ax_row[0].axis("off")

    # Col 1: B4 baseline
    ax_row[1].imshow(ov_b4b)
    ok = b4b_pred == true_label
    ax_row[1].set_title(f"{'✓' if ok else '✗'} {b4b_name}", fontsize=8, pad=4,
                         color="green" if ok else "red")
    ax_row[1].axis("off")

    # Col 2: B4 dendrite
    ax_row[2].imshow(ov_b4d)
    ok = b4d_pred == true_label
    ax_row[2].set_title(f"{'✓' if ok else '✗'} {b4d_name}", fontsize=8, pad=4,
                         color="green" if ok else "red")
    ax_row[2].axis("off")

    # Col 3: B5 baseline
    ax_row[3].imshow(ov_b5b)
    ok = b5b_pred == true_label
    ax_row[3].set_title(f"{'✓' if ok else '✗'} {b5b_name}", fontsize=8, pad=4,
                         color="green" if ok else "red")
    ax_row[3].axis("off")


def build_balanced_gradcam_figure(cases, b4_baseline, b4_dendrite, b5_baseline,
                                   output_path, use_plusplus=False):
    """
    Build the main balanced GradCAM figure showing all four case types.

    Layout: rows are grouped by case type with a section label on the left.
    The column headers show model name and accuracy stats.
    GradCAM is targeted at the TRUE class for all panels.
    """
    type_labels = {
        "dendrite_wins": "Dendrite wins\n(B4 wrong,\ndendrite right)",
        "b4_wins":       "B4 wins\n(Dendrite wrong,\nB4 right)",
        "b5_wins":       "B5 wins\n(Dendrite wrong,\nB5 right)",
        "all_wrong":     "All wrong\n(shared\nfailure)",
    }
    type_colors = {
        "dendrite_wins": "#1D9E75",
        "b4_wins":       "#378ADD",
        "b5_wins":       "#D85A30",
        "all_wrong":     "#888780",
    }

    # Count total rows
    total_rows = sum(len(v) for v in cases.values())
    if total_rows == 0:
        print("No cases found for balanced figure.")
        return

    fig, axes = plt.subplots(total_rows, 4, figsize=(20, 6 * total_rows))
    if total_rows == 1:
        axes = [axes]

    # Column headers on the first row
    col_titles = [
        "Original Image\n(Ground Truth)",
        "B4 Baseline\n17.7M | 92.57% Top-1",
        "B4 + 1 Dendrite (PAI)\n24.2M | 95.09% Top-1",
        "B5 Baseline\n28.5M | 95.85% Top-1",
    ]
    for col_i, title in enumerate(col_titles):
        axes[0][col_i].set_title(title, fontsize=10, fontweight="bold", pad=10)

    row_idx = 0
    for case_type, case_list in cases.items():
        label = type_labels[case_type]
        color = type_colors[case_type]
        for img, true_label, b4b_pred, b4d_pred, b5b_pred in case_list:
            build_gradcam_row(
                axes[row_idx], img, true_label,
                b4b_pred, b4d_pred, b5b_pred,
                b4_baseline, b4_dendrite, b5_baseline,
                label, use_plusplus
            )
            # Add colored left margin label for case type
            axes[row_idx][0].set_ylabel(label, fontsize=8, color=color,
                                         rotation=0, labelpad=60, va="center")
            row_idx += 1

    fig.suptitle(
        "Balanced GradCAM Analysis: All Four Case Types\n"
        "GradCAM targeted at ground-truth class for all panels",
        fontsize=13, fontweight="bold", y=1.002
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved balanced GradCAM figure → {output_path}")


def build_3way_comparison_figure(samples, b4_baseline, b4_dendrite, b5_baseline,
                                  output_path, use_plusplus=False):
    """
    Standard 3-way random sample figure (same as gradcam_3way.py).
    Shows 6 evenly-spaced test images with all three models side by side.
    """
    n = len(samples)
    fig = plt.figure(figsize=(20, 6 * n))
    fig.suptitle(
        "EfficientNet GradCAM: B4 Baseline vs B4+Dendrite vs B5 Baseline\n"
        "Oxford Flowers 102  —  10 images randomly sampled (numpy seed=42)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(n, 4, figure=fig, hspace=0.55, wspace=0.12)

    col_titles = [
        "Original Image\n(Ground Truth)",
        "B4 Baseline\n17.7M | 92.57% Top-1",
        "B4 + 1 Dendrite (PAI)\n24.2M | 95.09% Top-1",
        "B5 Baseline\n28.5M | 95.85% Top-1",
    ]
    for col_i, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col_i])
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
        ax.axis("off")

    for row_i, (pil_img, true_label) in enumerate(samples):
        t_b4   = pil_to_tensor(pil_img, B4_IMG_SIZE)
        t_b5   = pil_to_tensor(pil_img, B5_IMG_SIZE)
        rgb_b4 = tensor_to_rgb_array(t_b4)
        rgb_b5 = tensor_to_rgb_array(t_b5)
        class_name = FLOWER_CLASSES[true_label] if true_label < len(FLOWER_CLASSES) else str(true_label)

        with torch.no_grad():
            b4b_l = b4_baseline(t_b4)
            b4d_l = b4_dendrite(t_b4)
            b5b_l = b5_baseline(t_b5)

        def pred_info(logits, lbl):
            pred = logits.argmax(1).item()
            conf = torch.softmax(logits, 1)[0, pred].item()
            name = FLOWER_CLASSES[pred] if pred < len(FLOWER_CLASSES) else str(pred)
            return pred, conf, name, pred == lbl

        b4b_pred, b4b_conf, b4b_name, b4b_ok = pred_info(b4b_l, true_label)
        b4d_pred, b4d_conf, b4d_name, b4d_ok = pred_info(b4d_l, true_label)
        b5b_pred, b5b_conf, b5b_name, b5b_ok = pred_info(b5b_l, true_label)

        cam_b4b = run_gradcam(b4_baseline, t_b4, true_label, use_plusplus)
        cam_b4d = run_gradcam(b4_dendrite, t_b4, true_label, use_plusplus)
        cam_b5b = run_gradcam(b5_baseline, t_b5, true_label, use_plusplus)

        ov_b4b = show_cam_on_image(rgb_b4, cam_b4b, use_rgb=True)
        ov_b4d = show_cam_on_image(rgb_b4, cam_b4d, use_rgb=True)
        ov_b5b = np.array(
            Image.fromarray(show_cam_on_image(rgb_b5, cam_b5b, use_rgb=True))
            .resize((B4_IMG_SIZE, B4_IMG_SIZE))
        )

        ax0 = fig.add_subplot(gs[row_i, 0])
        ax0.imshow(pil_img.resize((B4_IMG_SIZE, B4_IMG_SIZE)))
        ax0.set_title(f"True: {class_name}", fontsize=10, pad=6)
        ax0.axis("off")

        for ax, ov, pred, conf, name, ok in [
            (fig.add_subplot(gs[row_i, 1]), ov_b4b, b4b_pred, b4b_conf, b4b_name, b4b_ok),
            (fig.add_subplot(gs[row_i, 2]), ov_b4d, b4d_pred, b4d_conf, b4d_name, b4d_ok),
            (fig.add_subplot(gs[row_i, 3]), ov_b5b, b5b_pred, b5b_conf, b5b_name, b5b_ok),
        ]:
            ax.imshow(ov)
            ax.set_title(f"{'✓' if ok else '✗'} {name}\n{conf:.1%}",
                         fontsize=10, pad=6, color="green" if ok else "red")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved 3-way comparison → {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
# Orchestrates the full pipeline:
#   1. Parse arguments
#   2. Download/locate checkpoints
#   3. Load all three models
#   4. Run full test set predictions
#   5. Run all statistical tests and generate report
#   6. Run GradCAM figures (if --run_gradcam flag is set)
#
# The --run_gradcam flag is optional because running GradCAM on many images
# takes significant GPU time. The statistical analysis (Section 5) runs on
# the full test set using only forward passes, which is much faster.
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Balanced statistical analysis + GradCAM for 3-way model comparison"
    )
    p.add_argument("--data_root",        default="/ocean/projects/cis260045p/shared/data")
    p.add_argument("--artifact_dir",     default="./wandb_artifacts")
    p.add_argument("--output_dir",       default="./gradcam_outputs_balanced")
    p.add_argument("--b4_baseline_ckpt", default=None,
                   help="Path to B4 baseline best_model.pt (skips WandB download)")
    p.add_argument("--dendrite_pai_dir", default=None,
                   help="Path to PAI artifacts dir for B4 dendrite model")
    p.add_argument("--b5_baseline_ckpt", default=None,
                   help="Path to B5 baseline best_model.pt (skips WandB download)")
    p.add_argument("--seed",             type=int, default=42,
                   help="Random seed for comparison figure sampling (default 42)")
    p.add_argument("--max_eval_images",  type=int, default=None,
                   help="Limit test set evaluation (None = full set ~6149 images)")
    p.add_argument("--run_gradcam",      action="store_true",
                   help="Also run GradCAM figures (slower, requires more GPU time)")
    p.add_argument("--use_plusplus",     action="store_true",
                   help="Use GradCAM++ instead of GradCAM (sharper heatmaps)")
    p.add_argument("--cases_per_type",   type=int, default=3,
                   help="Number of examples per case type in balanced figure")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)

    # ── Resolve checkpoints ───────────────────────────────────────────────────
    if args.b4_baseline_ckpt:
        b4_baseline_ckpt = args.b4_baseline_ckpt
    else:
        b4_art = download_artifact(ARTIFACT_B4_BASELINE,
                                   os.path.join(args.artifact_dir, "b4_baseline"))
        b4_baseline_ckpt = find_pt_file(b4_art)

    if args.dendrite_pai_dir:
        pai_dir = args.dendrite_pai_dir
    else:
        pai_dir = download_artifact(ARTIFACT_B4_DENDRITE,
                                    os.path.join(args.artifact_dir, PAI_SAVE_NAME))

    if args.b5_baseline_ckpt:
        b5_baseline_ckpt = args.b5_baseline_ckpt
    else:
        b5_art = download_artifact(ARTIFACT_B5_BASELINE,
                                   os.path.join(args.artifact_dir, "b5_baseline"))
        b5_baseline_ckpt = find_pt_file(b5_art)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n=== Loading B4 BASELINE ===")
    b4_baseline = build_b4_baseline(NUM_CLASSES)
    b4_baseline = load_standard_checkpoint(b4_baseline, b4_baseline_ckpt, "B4 Baseline")

    print("\n=== Loading B4 DENDRITE (PAI) ===")
    b4_dendrite = load_dendrite_model(pai_dir, NUM_CLASSES)

    print("\n=== Loading B5 BASELINE ===")
    b5_baseline = build_b5_baseline(NUM_CLASSES)
    b5_baseline = load_standard_checkpoint(b5_baseline, b5_baseline_ckpt, "B5 Baseline")

    # ── Statistical analysis (Section 5) ─────────────────────────────────────
    # Run all three models on the full test set first
    true_labels, b4b_preds, b4d_preds, b5b_preds, \
    b4b_correct, b4d_correct, b5b_correct = run_full_test_set_predictions(
        args.data_root, b4_baseline, b4_dendrite, b5_baseline,
        max_images=args.max_eval_images
    )

    # Run the full statistical test suite
    results = generate_statistical_report(
        true_labels, b4b_preds, b4d_preds, b5b_preds,
        b4b_correct, b4d_correct, b5b_correct,
        args.output_dir
    )

    # Per-class accuracy breakdown
    per_class_accuracy(true_labels, b4b_preds, b4d_preds, b5b_preds,
                       args.output_dir)

    # ── GradCAM figures (Section 7) — optional ────────────────────────────────
    if args.run_gradcam:
        print("\n=== Running GradCAM Figures ===")

        # Standard 3-way random sample figure
        samples = load_test_samples(args.data_root, NUM_SAMPLES, seed=args.seed)
        build_3way_comparison_figure(
            samples, b4_baseline, b4_dendrite, b5_baseline,
            os.path.join(args.output_dir, "gradcam_3way_comparison.png"),
            use_plusplus=args.use_plusplus,
        )

        # Balanced figure with all four case types
        cases = find_balanced_cases(
            args.data_root, b4_baseline, b4_dendrite, b5_baseline,
            cases_per_type=args.cases_per_type,
        )
        build_balanced_gradcam_figure(
            cases, b4_baseline, b4_dendrite, b5_baseline,
            os.path.join(args.output_dir, "gradcam_balanced_analysis.png"),
            use_plusplus=args.use_plusplus,
        )
    else:
        print("\n[Note] GradCAM figures skipped. Add --run_gradcam to generate them.")

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
