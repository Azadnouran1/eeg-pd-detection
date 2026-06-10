# =========================================================================
#  Grad-CAM — EWT / Custom Residual CNN / GroupKFold
#  ─────────────────────────────────────────────────────────────────────
#  STANDALONE — requires only:
#    (1) Saved .pth weights from the GroupKFold training run
#    (2) Saved image_predictions_groupkfold_fold*.csv files
#
#  DO NOT retrain. DO NOT rerun the original training script.
#  This script reads what is already on disk and produces figures.
#
#  OUTPUTS  (all written to GRADCAM_DIR):
#    Per-image panels  (.pdf, 3 columns: original | heatmap | overlay)
#      gradcam_EWT_<SubjectID>_<Outcome>_fold<N>.pdf
#
#    Cross-transform comparison figure  (.pdf, 3 rows × 4 columns)
#      gradcam_cross_transform_<SubjectID>.pdf
#      — Shows the same subject across EWT / CWT / STFT / DWT side-by-side.
#        Uses each transform's own fold model for that subject.
#        Requires that the subject appears as a validation subject in at
#        least one fold for each of the four transforms.
#
#  CRITICAL RULE — fold/model/image must always match:
#    An image from fold N's CSV must always be visualized with fold N's
#    .pth weights. The GroupKFold guarantee is that the validation subject
#    was never seen during that fold's training. Mixing folds breaks this.
# =========================================================================

import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


# =========================================================================
#  CONFIG — edit these paths to match your environment
# =========================================================================

# Base results directory for each transform
# The Custom Residual CNN results for EWT live directly under BASE_RESULTS/EWT/
BASE_RESULTS = "/scratch/user/u.aa346327/results"

# All four transforms — used for the cross-transform comparison figure
TRANSFORMS = ["EWT", "CWT", "STFT", "DWT"]

# Primary transform for the per-image panel figures (your best result)
PRIMARY_TAG = "EWT"

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SPLITS  = 5
IMG_SIZE  = 224

# How many images to select per outcome category (TP, TN, FP, FN)
# for the per-image panel figures
N_PER_OUTCOME = 3

# Output directory — all figures saved here
GRADCAM_DIR = os.path.join(BASE_RESULTS, PRIMARY_TAG, "gradcam")
os.makedirs(GRADCAM_DIR, exist_ok=True)

print(f"Device : {DEVICE}")
print(f"Output : {GRADCAM_DIR}")


# =========================================================================
#  MODEL DEFINITION
#  Copied verbatim from iowa_residual_cnn_v7.py — do not modify.
#  Must be identical to the architecture used during training so that
#  the saved .pth weights load correctly.
# =========================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, out_channels, 3,
                                  stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_channels)
        self.relu     = nn.ReLU(inplace=True)
        self.conv2    = nn.Conv2d(out_channels, out_channels, 3,
                                  padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResidualCustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                          # 224 → 112
        )
        self.res_block1 = ResidualBlock(32, 32)
        self.pool1      = nn.MaxPool2d(2)            # 112 → 56
        self.drop1      = nn.Dropout2d(0.1)

        self.res_block2 = ResidualBlock(32, 64)
        self.pool2      = nn.MaxPool2d(2)            # 56 → 28
        self.drop2      = nn.Dropout2d(0.1)

        self.res_block3 = ResidualBlock(64, 128)
        self.pool3      = nn.MaxPool2d(2)            # 28 → 14
        self.drop3      = nn.Dropout2d(0.1)

        self.gap = nn.AdaptiveAvgPool2d(1)           # 14 → 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.drop1(self.pool1(self.res_block1(x)))
        x = self.drop2(self.pool2(self.res_block2(x)))
        x = self.drop3(self.pool3(self.res_block3(x)))
        x = self.gap(x)
        return self.classifier(x)


def load_model(pth_path):
    """Load a saved fold model from disk. Always sets eval mode."""
    model = ResidualCustomCNN().to(DEVICE)
    model.load_state_dict(
        torch.load(pth_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model


# =========================================================================
#  VALIDATION TRANSFORM
#  Identical to val_transform in the original training script.
#  No augmentation — deterministic preprocessing only.
# =========================================================================

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# =========================================================================
#  GRAD-CAM
#  Target layer: model.res_block3.conv2
#    — last conv before pool3 → gap → classifier
#    — 128 channels, 14×14 spatial resolution
#    — highest-level spatial features before global pooling collapses them
#
#  Gradient direction: always w.r.t. the PD logit (positive output).
#    For a TP or FP image the logit is positive → gradient flows naturally.
#    For a TN or FN image the logit is negative → gradient still tells us
#    which regions pushed the model toward PD (even if not enough to cross
#    the 0.5 threshold). This is the correct and consistent choice for a
#    binary BCEWithLogitsLoss model.
# =========================================================================

class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.gradients   = None

        # Target: last conv in res_block3
        target = model.res_block3.conv2

        target.register_forward_hook(self._save_activations)
        target.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor):
        """
        Parameters
        ----------
        tensor : torch.Tensor  shape (1, 1, 224, 224)
            Single preprocessed image on DEVICE.
            Must be created with requires_grad=False on the tensor itself;
            we enable grad on the model parameters via the backward pass.

        Returns
        -------
        cam  : np.ndarray  shape (14, 14)  values in [0, 1]
        prob : float       sigmoid probability (PD score)
        pred : int         predicted class  0=HC  1=PD
        """
        self.model.eval()
        self.model.zero_grad()

        # Enable gradient computation for this forward pass
        with torch.enable_grad():
            logit = self.model(tensor)          # (1, 1)
            prob  = torch.sigmoid(logit).item()
            pred  = int(prob >= 0.5)

            # Backward w.r.t. PD logit (always positive direction)
            logit.backward()

        # Grad-CAM: channel weights = global average of gradients
        # gradients shape: (1, 128, 14, 14)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1,128,1,1)
        cam     = (weights * self.activations).sum(dim=1).squeeze()  # (14,14)
        cam     = F.relu(cam)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy(), prob, pred


def prepare_tensor(image_path):
    """Load an image from disk and convert to model-ready tensor on DEVICE."""
    img    = Image.open(image_path).convert("L")
    tensor = val_transform(img).unsqueeze(0).to(DEVICE)
    return tensor


def build_overlay(image_path, cam, alpha=0.45):
    """
    Returns three uint8 RGB arrays (H, W, 3):
      orig_rgb  — original spectrogram in RGB colour space
      heatmap   — jet colourmap applied to Grad-CAM
      blended   — weighted overlay of heatmap on original
    """
    orig     = np.array(Image.open(image_path).convert("L").resize((IMG_SIZE, IMG_SIZE)))
    cam_r    = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    heatmap  = (plt.cm.jet(cam_r)[:, :, :3] * 255).astype(np.uint8)
    orig_rgb = np.stack([orig] * 3, axis=-1)
    blended  = (alpha * heatmap + (1 - alpha) * orig_rgb).astype(np.uint8)
    return orig_rgb, heatmap, blended


# =========================================================================
#  IMAGE SELECTION
#  Reads the same CSVs that groupkfold_standard_recompute.py uses.
#  Image-level TP/TN/FP/FN — no subject aggregation, no majority voting.
# =========================================================================

OUTCOME_LABELS = {
    "TP": "PD — correctly classified",
    "TN": "HC — correctly classified",
    "FP": "HC — misclassified as PD",
    "FN": "PD — misclassified as HC",
}

def load_all_predictions(tag, base_results, n_splits=N_SPLITS):
    """
    Reads all five fold CSVs for a given transform tag.
    Returns a single DataFrame with an added 'Fold' column.
    """
    dfs = []
    for fold in range(1, n_splits + 1):
        csv_path = os.path.join(
            base_results, tag,
            f"{tag}_image_predictions_groupkfold_fold{fold}.csv"
        )
        if not os.path.isfile(csv_path):
            print(f"  [WARN] Missing CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["Fold"] = fold
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(
            f"No prediction CSVs found for tag={tag} under {base_results}"
        )
    return pd.concat(dfs, ignore_index=True)


def assign_outcome(df):
    """Add an 'Outcome' column (TP/TN/FP/FN) at image level."""
    conditions = [
        (df["True_Label"] == 1) & (df["Pred_Label"] == 1),
        (df["True_Label"] == 0) & (df["Pred_Label"] == 0),
        (df["True_Label"] == 0) & (df["Pred_Label"] == 1),
        (df["True_Label"] == 1) & (df["Pred_Label"] == 0),
    ]
    outcomes = ["TP", "TN", "FP", "FN"]
    df = df.copy()
    df["Outcome"] = np.select(conditions, outcomes, default="Unknown")
    return df


def select_images(all_df, n_per_outcome=N_PER_OUTCOME):
    """
    For each outcome category, selects up to n_per_outcome images.

    Selection strategy:
      TP  — highest Pred_Prob  (model most confidently correct on PD)
      TN  — lowest  Pred_Prob  (model most confidently correct on HC)
      FP  — highest Pred_Prob  (HC the model was most wrong about)
      FN  — lowest  Pred_Prob  (PD the model missed most confidently)

    One image per subject maximum to ensure diversity across subjects.
    """
    selected = {}
    for outcome in ["TP", "TN", "FP", "FN"]:
        subset = all_df[all_df["Outcome"] == outcome].copy()
        if subset.empty:
            print(f"  [WARN] No {outcome} images found.")
            selected[outcome] = pd.DataFrame()
            continue

        ascending = outcome in ["TN", "FN"]
        subset = subset.sort_values("Pred_Prob", ascending=ascending)

        # One image per subject — keeps visual diversity
        subset = subset.drop_duplicates(subset=["Subject_ID"])
        selected[outcome] = subset.head(n_per_outcome)

    return selected


# =========================================================================
#  FIGURE 1 — Per-image 3-panel figure
#  One PDF per selected image.
#  Layout: [Original EWT Spectrogram | Grad-CAM Heatmap | Overlay]
# =========================================================================

def save_panel_figure(image_path, cam, prob, pred,
                      subject_id, true_label, outcome,
                      fold, tag, save_path):
    orig_rgb, heatmap, blended = build_overlay(image_path, cam)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    panel_titles = [
        f"{tag} Spectrogram",
        "Grad-CAM Activation Map",
        "Overlay"
    ]
    images = [orig_rgb, heatmap, blended]

    for ax, img, title in zip(axes, images, panel_titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, pad=8)
        ax.axis("off")

    class_names = {0: "HC", 1: "PD"}
    suptitle = (
        f"Subject: {subject_id}  |  True: {class_names[true_label]}  |  "
        f"Predicted: {class_names[pred]}  |  PD probability: {prob:.3f}\n"
        f"Outcome: {OUTCOME_LABELS[outcome]}  |  "
        f"Fold: {fold}  |  Layer: res_block3.conv2"
    )
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)

    plt.tight_layout()
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {os.path.basename(save_path)}")


def run_per_image_figures(tag, base_results, gradcam_dir,
                          n_per_outcome=N_PER_OUTCOME):
    """
    Main driver for Figure 1.
    Reads CSVs → selects images → loads correct fold model → runs Grad-CAM
    → saves one PDF panel per image.
    """
    print(f"\n{'='*60}")
    print(f"  Figure 1 — Per-image panels  [{tag}]")
    print(f"{'='*60}")

    all_df   = load_all_predictions(tag, base_results)
    all_df   = assign_outcome(all_df)
    selected = select_images(all_df, n_per_outcome)

    # Cache loaded models to avoid reloading the same fold model repeatedly
    model_cache = {}

    for outcome, subset in selected.items():
        if subset.empty:
            continue
        print(f"\n  Outcome: {outcome}  ({len(subset)} images)")

        for _, row in subset.iterrows():
            fold       = int(row["Fold"])
            image_path = row["Image_Path"]
            subject_id = row["Subject_ID"]
            true_label = int(row["True_Label"])

            # Verify image exists on disk
            if not os.path.isfile(image_path):
                print(f"    [SKIP] Image not found on disk: {image_path}")
                continue

            # Load the fold model (cached)
            if fold not in model_cache:
                pth_path = os.path.join(
                    base_results, tag, "models",
                    f"{tag}_HC_vs_PD_groupkfold_fold{fold}_best.pth"
                )
                if not os.path.isfile(pth_path):
                    print(f"    [SKIP] Model not found: {pth_path}")
                    continue
                model_cache[fold] = load_model(pth_path)
                print(f"    Loaded fold {fold} model")

            model = model_cache[fold]
            gcam  = GradCAM(model)

            tensor          = prepare_tensor(image_path)
            cam, prob, pred = gcam.generate(tensor)

            fname = (
                f"gradcam_{tag}_{subject_id}_{outcome}_fold{fold}.pdf"
            )
            save_path = os.path.join(gradcam_dir, fname)

            save_panel_figure(
                image_path, cam, prob, pred,
                subject_id, true_label, outcome,
                fold, tag, save_path
            )

    print(f"\n  Figure 1 complete. Files in: {gradcam_dir}")


# =========================================================================
#  FIGURE 2 — Cross-transform comparison (3 rows × 4 columns)
#  Shows the SAME subject across EWT / CWT / STFT / DWT.
#  Rows: Original | Grad-CAM | Overlay
#  Columns: EWT | CWT | STFT | DWT
#
#  Subject selection: picks a TP subject from EWT (your best result)
#  that also appears as a validation subject in at least one fold
#  for each of the other three transforms. Falls back gracefully if
#  a perfect match is not available.
# =========================================================================

def find_subject_fold(subject_id, tag, base_results, n_splits=N_SPLITS):
    """
    Given a subject_id, finds the fold in which that subject was in
    the validation set for the given transform tag.
    Returns (fold, representative_image_path) or (None, None).

    Representative image = the image whose Pred_Prob is closest to
    the subject's mean Pred_Prob across all their validation images.
    This avoids cherry-picking an outlier activation.
    """
    for fold in range(1, n_splits + 1):
        csv_path = os.path.join(
            base_results, tag,
            f"{tag}_image_predictions_groupkfold_fold{fold}.csv"
        )
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        subj_df = df[df["Subject_ID"] == subject_id]
        if subj_df.empty:
            continue

        mean_prob = subj_df["Pred_Prob"].mean()
        rep_idx   = (subj_df["Pred_Prob"] - mean_prob).abs().idxmin()
        rep_path  = subj_df.loc[rep_idx, "Image_Path"]
        return fold, rep_path

    return None, None


def pick_cross_transform_subject(primary_tag, base_results, transforms,
                                 preferred_outcome="TP"):
    """
    Finds a subject that:
      (a) Is a validation subject in primary_tag with preferred_outcome
      (b) Also appears as a validation subject in every other transform

    Returns subject_id and a dict {tag: (fold, image_path)}.
    Falls back to any outcome if preferred_outcome not available.
    """
    primary_df = load_all_predictions(primary_tag, base_results)
    primary_df = assign_outcome(primary_df)

    # Try preferred outcome first, then fall back to any correct prediction
    for fallback_outcome in [preferred_outcome, "TN", "FP", "FN"]:
        candidates = primary_df[
            primary_df["Outcome"] == fallback_outcome
        ]["Subject_ID"].unique()

        for subject_id in candidates:
            fold_map = {}
            found_all = True

            for tag in transforms:
                fold, img_path = find_subject_fold(
                    subject_id, tag, base_results
                )
                if fold is None or not os.path.isfile(img_path):
                    found_all = False
                    break
                fold_map[tag] = (fold, img_path)

            if found_all:
                print(
                    f"  Cross-transform subject: {subject_id} "
                    f"(primary outcome: {fallback_outcome})"
                )
                return subject_id, fold_map

    print("  [WARN] No subject found in validation for all 4 transforms.")
    return None, None


def save_cross_transform_figure(subject_id, fold_map, base_results,
                                transforms, gradcam_dir):
    """
    Produces the 3-row × 4-column journal figure.
    Each column uses the fold model specific to that transform and fold.
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    row_labels = ["Original Spectrogram", "Grad-CAM Activation", "Overlay"]
    col_probs  = []

    for col, tag in enumerate(transforms):
        fold, img_path = fold_map[tag]

        # Load this transform's fold model
        pth_path = os.path.join(
            base_results, tag, "models",
            f"{tag}_HC_vs_PD_groupkfold_fold{fold}_best.pth"
        )
        if not os.path.isfile(pth_path):
            print(f"  [SKIP column] Model not found: {pth_path}")
            for row in range(3):
                axes[row][col].axis("off")
                axes[row][col].set_title(f"{tag}\n(model missing)", fontsize=11)
            continue

        model = load_model(pth_path)
        gcam  = GradCAM(model)

        tensor          = prepare_tensor(img_path)
        cam, prob, pred = gcam.generate(tensor)
        col_probs.append(prob)

        orig_rgb, heatmap, blended = build_overlay(img_path, cam)

        for row, (img, row_label) in enumerate(
            zip([orig_rgb, heatmap, blended], row_labels)
        ):
            ax = axes[row][col]
            ax.imshow(img)
            ax.axis("off")

            if row == 0:
                ax.set_title(
                    f"{tag}\nPD prob: {prob:.3f}  |  Fold {fold}",
                    fontsize=12, fontweight="bold", pad=6
                )
            if col == 0:
                ax.set_ylabel(row_label, fontsize=11, labelpad=8)

    fig.suptitle(
        f"Grad-CAM Across Time-Frequency Representations\n"
        f"Subject: {subject_id}  |  Layer: res_block3.conv2",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    save_path = os.path.join(
        gradcam_dir,
        f"gradcam_cross_transform_{subject_id}.pdf"
    )
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


def run_cross_transform_figure(transforms, base_results, gradcam_dir):
    """
    Main driver for Figure 2.
    Picks the best cross-transform subject automatically and saves the figure.
    """
    print(f"\n{'='*60}")
    print(f"  Figure 2 — Cross-transform comparison")
    print(f"{'='*60}")

    subject_id, fold_map = pick_cross_transform_subject(
        primary_tag  = PRIMARY_TAG,
        base_results = base_results,
        transforms   = transforms,
        preferred_outcome = "TP"
    )

    if subject_id is None:
        print("  [SKIP] Cross-transform figure skipped — no valid subject found.")
        return

    save_cross_transform_figure(
        subject_id, fold_map, base_results, transforms, gradcam_dir
    )
    print(f"\n  Figure 2 complete.")


# =========================================================================
#  FIGURE 3 — Summary grid (all selected images in one figure)
#  Layout: rows = one per selected image, columns = outcome category
#  Useful for the supplementary material section of the paper.
# =========================================================================

def run_summary_grid(tag, base_results, gradcam_dir,
                     n_per_outcome=N_PER_OUTCOME):
    """
    Produces one large PDF grid showing all selected images together.
    Columns: TP | TN | FP | FN
    Rows:    up to n_per_outcome per column
    Each cell: overlay image only (compact for supplementary)
    """
    print(f"\n{'='*60}")
    print(f"  Figure 3 — Summary grid  [{tag}]")
    print(f"{'='*60}")

    all_df   = load_all_predictions(tag, base_results)
    all_df   = assign_outcome(all_df)
    selected = select_images(all_df, n_per_outcome)

    outcomes    = ["TP", "TN", "FP", "FN"]
    n_rows      = n_per_outcome
    n_cols      = len(outcomes)
    fig, axes   = plt.subplots(n_rows, n_cols,
                               figsize=(5 * n_cols, 5 * n_rows))

    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    model_cache = {}

    for col, outcome in enumerate(outcomes):
        subset = selected.get(outcome, pd.DataFrame())
        axes[0][col].set_title(
            OUTCOME_LABELS[outcome], fontsize=11,
            fontweight="bold", pad=8
        )

        for row in range(n_rows):
            ax = axes[row][col]
            ax.axis("off")

            if subset.empty or row >= len(subset):
                continue

            record     = subset.iloc[row]
            fold       = int(record["Fold"])
            image_path = record["Image_Path"]
            subject_id = record["Subject_ID"]
            prob_val   = float(record["Pred_Prob"])

            if not os.path.isfile(image_path):
                continue

            if fold not in model_cache:
                pth_path = os.path.join(
                    base_results, tag, "models",
                    f"{tag}_HC_vs_PD_groupkfold_fold{fold}_best.pth"
                )
                if not os.path.isfile(pth_path):
                    continue
                model_cache[fold] = load_model(pth_path)

            model           = model_cache[fold]
            gcam            = GradCAM(model)
            tensor          = prepare_tensor(image_path)
            cam, prob, pred = gcam.generate(tensor)

            _, _, blended = build_overlay(image_path, cam)
            ax.imshow(blended)
            ax.set_title(
                f"{subject_id}\np={prob:.3f}  fold {fold}",
                fontsize=9
            )

    fig.suptitle(
        f"Grad-CAM Summary Grid — {tag} / Custom Residual CNN / GroupKFold",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_path = os.path.join(gradcam_dir, f"gradcam_{tag}_summary_grid.pdf")
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")
    print(f"\n  Figure 3 complete.")


# =========================================================================
#  ENTRY POINT
# =========================================================================

if __name__ == "__main__":

    print("\n" + "#" * 70)
    print("#  Grad-CAM — Custom Residual CNN / GroupKFold / Standard Metrics")
    print("#" * 70)

    # ── Figure 1: per-image 3-panel PDFs for EWT (your best result) ──────
    run_per_image_figures(
        tag          = PRIMARY_TAG,
        base_results = BASE_RESULTS,
        gradcam_dir  = GRADCAM_DIR,
        n_per_outcome = N_PER_OUTCOME,
    )

    # ── Figure 2: cross-transform comparison (EWT vs CWT vs STFT vs DWT) ─
    run_cross_transform_figure(
        transforms   = TRANSFORMS,
        base_results = BASE_RESULTS,
        gradcam_dir  = GRADCAM_DIR,
    )

    # ── Figure 3: compact summary grid for supplementary material ─────────
    run_summary_grid(
        tag          = PRIMARY_TAG,
        base_results = BASE_RESULTS,
        gradcam_dir  = GRADCAM_DIR,
        n_per_outcome = N_PER_OUTCOME,
    )

    print("\n" + "#" * 70)
    print("#  ALL FIGURES COMPLETE")
    print(f"#  Output directory: {GRADCAM_DIR}")
    print("#" * 70)
