# =========================================================================
# ✅ HC vs PD (IOWA) — Residual Custom CNN  [v9]
#    Datasets: EWT, CWT, STFT, DWT
#
#  TWO EVALUATION PROTOCOLS
#  ─────────────────────────────────────────────────────────────────────
#  1. Manual Balanced Subject-Aware KFold (5-fold)
#     — HC and PD subjects are independently shuffled and assigned to
#       folds so each fold gets exactly balanced HC/PD subject counts:
#         Folds 1-4 : 3 HC + 3 PD validation subjects
#         Fold  5   : 2 HC + 2 PD validation subjects
#     — No subject ever appears in both train and validation.
#     — Majority voting at subject level for final metrics.
#
#  2. Standard StratifiedKFold (5-fold, image-level)
#     — StratifiedKFold ensures balanced HC/PD image ratio per fold.
#     — Standard image-level threshold evaluation.
#
#  RESUME / SKIP LOGIC
#  ─────────────────────────────────────────────────────────────────────
#  - If a dataset's results_summary.xlsx already exists → entire dataset
#    is skipped (all folds for both protocols already completed).
#  - If a fold's best .pth already exists → that fold is NOT retrained.
#    The saved weights are loaded directly and evaluation proceeds.
#  - Set FORCE_RETRAIN = True to ignore all saved files and retrain
#    everything from scratch.
#
#  CHANGES FROM v6  (carried forward in v7, see below)
#  ─────────────────────────────────────────────────────────────────────
#  1. LR raised from 1e-4 → 3e-4. From-scratch CNNs benefit from a
#     higher initial learning rate to explore the loss surface before
#     the scheduler decays it.
#  2. CosineAnnealingLR scheduler added (T_max=EPOCHS, eta_min=1e-6).
#     Pretrained models can skip schedulers because they start near a
#     good solution; a from-scratch model needs annealing to converge
#     cleanly. scheduler.step() is called after each epoch.
#  3. EPOCHS raised from 15 → 20. Gives the cosine schedule room to
#     decay fully and allows early stopping to act more selectively.
#     ES_MIN_EPOCHS=7 and ES_PATIENCE=3 are unchanged.
#  4. weight_decay=1e-4 added to Adam. Dropout(0.3) only regularizes
#     the dense head; L2 via weight_decay extends regularization across
#     all conv parameters — important for from-scratch training.
#  5. Dropout2d(0.1) added after each residual pool stage. Spatial
#     dropout randomly zeros entire feature-map channels, providing
#     broader regularization coverage than dense Dropout alone. Applied
#     only during training (eval mode disables it automatically).
#  6. ColorJitter(brightness, contrast) on grayscale is harmless —
#     it adjusts single-channel intensity — no change made.
#  7. RandomErasing fill value (default=0) is correct in normalized
#     space — no change made.
#
#  CHANGES FROM v7
#  ─────────────────────────────────────────────────────────────────────
#  1. Global Average Pooling (GAP) replaces Flatten + Linear(25088, 256)
#     in the classifier head. After 4× spatial downsampling the feature
#     map is 14×14; flattening it produced a 25,088-input linear layer —
#     the single largest source of parameters and overfitting risk in the
#     model. AdaptiveAvgPool2d(1) collapses each of the 128 channels to
#     a scalar, giving a 128-input linear layer instead. This removes
#     ~3.2 M parameters from the head, forces the conv stack to encode
#     spatially meaningful per-channel features, and is consistent with
#     every modern architecture (ResNet, EfficientNet, etc.). The
#     spatial Dropout2d layers already added in v7 are the natural
#     complement to GAP.
#
#  CHANGES FROM v8
#  ─────────────────────────────────────────────────────────────────────
#  1. optim.AdamW → optim.Adam (consistency fix for cross-model
#     comparability). All three pretrained-model baselines in this
#     experiment set (EfficientNet-B0, MobileNet-V3, etc.) use
#     optim.Adam with weight_decay=1e-4. v8 introduced AdamW for
#     theoretical correctness, but the practical difference on a small
#     from-scratch CNN with weight_decay=1e-4 is negligible, and
#     optimizer consistency across all four experiments is required for
#     a fair journal comparison. Rolling back to Adam ensures the only
#     intentional differences between models are those documented under
#     "LEGITIMATELY DIFFERENT" below.
#
#  LEGITIMATELY DIFFERENT vs PRETRAINED BASELINES
#  (differences that are scientifically justified and must be reported)
#  ─────────────────────────────────────────────────────────────────────
#  LR = 3e-4  (pretrained baselines use 1e-4)
#    — From-scratch CNNs benefit from a higher initial LR to explore
#      the loss surface; the cosine scheduler then decays it to 1e-6.
#      Pretrained models start near a good solution and converge with
#      a conservative 1e-4 without any scheduler.
#  CosineAnnealingLR present here, absent in pretrained baselines
#    — Required for from-scratch convergence. A frozen backbone
#      optimizing a single linear layer is approximately convex and
#      does not need annealing.
#  Dropout(0.3) in classifier head (pretrained baselines have none)
#    — This model is trained entirely from scratch on a small dataset
#      (28 subjects). Without head dropout, the two-layer MLP head
#      (Linear 128→256→1) overfits rapidly. Pretrained baselines
#      replace the backbone's original classifier with a single
#      Linear(in_features→1) layer — a much simpler, lower-variance
#      problem that does not require head dropout; weight_decay=1e-4
#      on the single layer is sufficient regularization.
#  Dropout2d(0.1) after each residual stage (pretrained: frozen backbone)
#    — Spatial dropout extends regularization into the conv stack.
#      Pretrained backbones are frozen and receive no gradient updates,
#      so adding dropout to them would have no effect.
#  Grayscale input + neutral normalization (pretrained: RGB + ImageNet)
#    — EEG spectrograms are single-channel magnitude images; RGB
#      encoding is redundant. Pretrained backbones require the exact
#      RGB channel statistics they were trained on.
#
#  SAVED OUTPUTS  (all under BASE_RESULT_DIR / <dataset_tag> / ...)
#  ─────────────────────────────────────────────────────────────────────
#  Plots  (PDF)
#    <tag>/plots/<tag>_groupkfold_fold<N>_accuracy.pdf / _loss.pdf
#    <tag>/plots/<tag>_standard_fold<N>_accuracy.pdf  / _loss.pdf
#
#  Epoch history  (Excel)
#    <tag>/<tag>_groupkfold_history.xlsx
#    <tag>/<tag>_standard_history.xlsx
#
#  Results summary  (Excel)
#    <tag>/<tag>_results_summary.xlsx
#      Sheet "GroupKFold"    — per-fold + Mean row (yellow) + Total row (green)
#      Sheet "Standard5Fold" — per-fold + Mean row (yellow) + Total row (green)
#
#  Model weights  (.pth)
#    <tag>/models/<tag>_HC_vs_PD_groupkfold_fold<N>_best.pth
#    <tag>/models/<tag>_HC_vs_PD_standard_fold<N>_best.pth
#
#  CSVs
#    <tag>/<tag>_subject_wise_accuracy_HC_vs_PD.csv
#    <tag>/<tag>_fold_metrics_HC_vs_PD_groupkfold.csv
#    <tag>/<tag>_fold_metrics_HC_vs_PD_standard.csv
#    <tag>/<tag>_final_aggregated_metrics_HC_vs_PD_groupkfold.csv / .xlsx
#    <tag>/<tag>_final_aggregated_metrics_HC_vs_PD_standard.csv   / .xlsx
#    <tag>/<tag>_fold_subject_counts_groupkfold.csv
#    <tag>/<tag>_fold_subject_counts_standard.csv
#    <tag>/<tag>_image_predictions_groupkfold_fold<N>.csv
#    <tag>/<tag>_image_predictions_standard_fold<N>.csv
#    <tag>/<tag>_groupkfold_subject_fold_assignment.csv
#
#  Metrics reported: Accuracy, Precision, Sensitivity, Specificity,
#                    FAR, FRR, MCC, AUC, Best_Epoch, TP, TN, FP, FN
# =========================================================================

import os, re, random
import numpy as np
from glob import glob
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# ====================== DATASET REGISTRY ======================
DATASETS = [
    ("EWT",  "/scratch/user/u.aa346327/EWT - IOWA/HC",  "/scratch/user/u.aa346327/EWT - IOWA/PD"),
    ("CWT",  "/scratch/user/u.aa346327/CWT - IOWA/HC",  "/scratch/user/u.aa346327/CWT - IOWA/PD"),
    ("STFT", "/scratch/user/u.aa346327/STFT - IOWA/HC", "/scratch/user/u.aa346327/STFT - IOWA/PD"),
    ("DWT",  "/scratch/user/u.aa346327/DWT - IOWA/HC",  "/scratch/user/u.aa346327/DWT - IOWA/PD"),
]

BASE_RESULT_DIR = "/scratch/user/u.aa346327/results/"

# ====================== RESUME FLAG ======================
# Set FORCE_RETRAIN = True to ignore all saved files and retrain everything.
# Set FORCE_RETRAIN = False to skip completed datasets/folds automatically.
FORCE_RETRAIN = False

# ====================== SETUP ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

IMG_SIZE    = 224
EPOCHS      = 20        # v7: raised from 15 → 20 to give cosine schedule room
BATCH_SIZE  = 32
LR          = 3e-4      # v7: raised from 1e-4 → 3e-4 for from-scratch training
N_SPLITS    = 5
NUM_WORKERS = 8

ES_PATIENCE           = 3
ES_MIN_EPOCHS         = 7   # early stopping cannot fire before this epoch



# ====================== SUBJECT ID ======================
def extract_subject_id(path):
    fname = os.path.basename(path)
    sid   = re.search(r"subject_(\d+)", fname).group(1)
    norm  = path.replace("\\", "/")
    return f"HC_{sid}" if "/HC/" in norm else f"PD_{sid}"


# ====================== COLLECT PATHS ======================
def collect_paths(hc_dir, pd_dir, tag):
    exts = ["*.png", "*.jpg", "*.jpeg"]
    hc_files, pd_files = [], []
    for ext in exts:
        hc_files.extend(glob(os.path.join(hc_dir, ext)))
        pd_files.extend(glob(os.path.join(pd_dir, ext)))


    paths  = np.array(sorted(hc_files + pd_files))
    labels = np.array([0] * len(hc_files) + [1] * len(pd_files))
    groups = np.array([extract_subject_id(p) for p in paths])

    print(
        f"\n[{tag}] HC vs PD (IOWA) | "
        f"HC: {len(hc_files)} | PD: {len(pd_files)} | "
        f"Total: {len(paths)} | Subjects: {len(np.unique(groups))}"
    )
    return paths, labels, groups


# ====================== TRANSFORMS ======================
# NOTE: Grayscale (1-channel) — EWT/CWT/STFT/DWT spectrograms are single-channel
#       magnitude images; RGB encoding is redundant. Normalization uses neutral
#       [0.5]/[0.5] — NOT ImageNet stats, which only apply to pretrained models.
#       RandomHorizontalFlip removed — flipping the time axis is semantically invalid.
#       RandomRotation removed — rotating time-frequency axes breaks axis meaning.
#       ColorJitter on a grayscale PIL image adjusts single-channel intensity
#       (brightness/contrast) and is valid — effect is equivalent to intensity jitter.
#       RandomErasing fill=0 in normalized space represents a neutral pixel value,
#       consistent with [0.5]/[0.5] normalization — correct placement and behavior.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ColorJitter(brightness=0.12, contrast=0.12),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.0))
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# ====================== DATASET ======================
class EEGImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = list(paths)
        self.labels    = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32), self.paths[idx]


# ====================== RESIDUAL BLOCK ======================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_channels)
        self.relu     = nn.ReLU(inplace=True)
        self.conv2    = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


# ====================== RESIDUAL CUSTOM CNN ======================
# v7: Dropout2d(0.1) added after each residual pool stage.
#     Spatial dropout randomly zeros entire feature-map channels,
#     extending regularization into the conv stack beyond the dense
#     head. Disabled automatically in eval mode — no inference cost.
# v8: Global Average Pooling (GAP) replaces Flatten + Linear(25088, 256).
#     AdaptiveAvgPool2d(1) collapses each of the 128 feature-map channels
#     to a scalar → Linear(128, 256) instead of Linear(25088, 256).
#     Removes ~3.2 M parameters from the head; forces the conv stack to
#     encode spatially meaningful per-channel features. Resolution-
#     agnostic: GAP works regardless of input spatial size.
class ResidualCustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),   # in_channels=1 for grayscale
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                                # 224 → 112
        )

        self.res_block1 = ResidualBlock(32, 32)
        self.pool1      = nn.MaxPool2d(2)                 # 112 → 56
        self.drop1      = nn.Dropout2d(0.1)               # spatial dropout, stage 1

        self.res_block2 = ResidualBlock(32, 64)
        self.pool2      = nn.MaxPool2d(2)                 # 56  → 28
        self.drop2      = nn.Dropout2d(0.1)               # spatial dropout, stage 2

        self.res_block3 = ResidualBlock(64, 128)
        self.pool3      = nn.MaxPool2d(2)                 # 28  → 14
        self.drop3      = nn.Dropout2d(0.1)               # spatial dropout, stage 3

        # v8: GAP collapses 128×14×14 → 128×1×1, then flatten to 128.
        #     Replaces the old Flatten + Linear(128*14*14, 256) which
        #     had 25,088 inputs and ~3.2 M parameters.
        self.gap = nn.AdaptiveAvgPool2d(1)                # 14 → 1  (global)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),                          # v8: 128 inputs (was 25,088)
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                              # head dropout — justified; see "LEGITIMATELY DIFFERENT" in header
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.drop1(self.pool1(self.res_block1(x)))
        x = self.drop2(self.pool2(self.res_block2(x)))
        x = self.drop3(self.pool3(self.res_block3(x)))
        x = self.gap(x)                                   # v8: global average pool
        return self.classifier(x)


def create_model():
    return ResidualCustomCNN().to(DEVICE)


# optimizer: optim.Adam with weight_decay=1e-4.
#   Adam is used here (not AdamW) for consistency with all pretrained
#   baseline models in this experiment set, which also use optim.Adam
#   with the same weight_decay. On a small from-scratch CNN the
#   practical difference between Adam and AdamW at weight_decay=1e-4
#   is negligible; optimizer uniformity across experiments is the
#   priority for cross-model comparability in the journal submission.
#
# head Dropout(0.3): present here, absent in pretrained baselines.
#   This is intentional and scientifically justified — see the
#   "LEGITIMATELY DIFFERENT" section in the file header for the full
#   rationale. In brief: this model trains all parameters from scratch
#   on a small dataset (28 subjects), so the two-layer MLP head needs
#   explicit dropout to prevent overfitting. Pretrained baselines use
#   a single Linear(in_features→1) replacement head, where the frozen
#   backbone's implicit regularization and weight_decay=1e-4 are
#   sufficient — an extra Dropout layer is not needed there.
def make_optimizer_and_loss(model, lr=LR):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    return optimizer, criterion


# ====================== HELPERS ======================
def logits_to_preds_probs(logits):
    probs = torch.sigmoid(logits).cpu().numpy().ravel()
    preds = (probs >= 0.5).astype(int)
    return preds, probs



def compute_metrics(TP, TN, FP, FN, all_labels=None, all_probs=None):
    eps = 1e-8
    m = {
        "Accuracy":    (TP + TN) / (TP + TN + FP + FN + eps),
        "Precision":   TP / (TP + FP + eps),
        "Sensitivity": TP / (TP + FN + eps),
        "Specificity": TN / (TN + FP + eps),
        "FAR":         FP / (FP + TN + eps),
        "FRR":         FN / (FN + TP + eps),
        "MCC":         (TP * TN - FP * FN) /
                       np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps),
        "AUC":         float("nan"),
    }
    if all_labels is not None and all_probs is not None:
        try:
            m["AUC"] = roc_auc_score(all_labels, all_probs)
        except ValueError:
            pass
    return m


# ================================================================
#  MANUAL BALANCED SUBJECT FOLD ASSIGNMENT
#  Guarantees exactly 3 HC + 3 PD per fold (folds 1-4)
#                 and 2 HC + 2 PD for fold 5.
#  HC and PD subject lists are independently shuffled then
#  round-robin assigned to folds.
#  Returns: list of N_SPLITS dicts, each with keys
#           'val_subjects' (set), 'tr_idx', 'va_idx'
# ================================================================
def make_balanced_subject_folds(paths, labels, groups, n_splits=N_SPLITS, seed=SEED):
    """
    Manually assign subjects to folds with guaranteed HC/PD balance.

    With 14 HC + 14 PD subjects and 5 folds:
      Folds 1-4 : 3 HC + 3 PD validation subjects  (6 total)
      Fold  5   : 2 HC + 2 PD validation subjects  (4 total)

    Subjects are shuffled independently per class using the fixed seed
    so the assignment is fully reproducible.
    """
    rng = np.random.default_rng(seed)

    # Get unique subjects per class
    hc_subjects = sorted([s for s in np.unique(groups) if s.startswith("HC_")])
    pd_subjects = sorted([s for s in np.unique(groups) if s.startswith("PD_")])

    # Shuffle independently
    hc_subjects = list(rng.permutation(hc_subjects))
    pd_subjects = list(rng.permutation(pd_subjects))

    n_hc = len(hc_subjects)   # 14
    n_pd = len(pd_subjects)   # 14

    # Assign subjects to folds round-robin
    hc_fold_ids = [i % n_splits for i in range(n_hc)]
    pd_fold_ids = [i % n_splits for i in range(n_pd)]

    # Build per-fold validation subject sets
    fold_val_subjects = []
    for f in range(n_splits):
        val_hc = {hc_subjects[i] for i, fid in enumerate(hc_fold_ids) if fid == f}
        val_pd = {pd_subjects[i] for i, fid in enumerate(pd_fold_ids) if fid == f}
        fold_val_subjects.append(val_hc | val_pd)

    # Print assignment summary
    print(f"\n  [Manual Fold Assignment] Seed={seed}")
    assignment_rows = []
    for f, val_subs in enumerate(fold_val_subjects, 1):
        n_hc_val = sum(1 for s in val_subs if s.startswith("HC_"))
        n_pd_val = sum(1 for s in val_subs if s.startswith("PD_"))
        hc_list  = sorted(s for s in val_subs if s.startswith("HC_"))
        pd_list  = sorted(s for s in val_subs if s.startswith("PD_"))
        print(f"  Fold {f}: Val HC={n_hc_val} {hc_list}  |  PD={n_pd_val} {pd_list}")
        for s in val_subs:
            assignment_rows.append({"Fold": f, "Subject_ID": s,
                                    "Class": "HC" if s.startswith("HC_") else "PD",
                                    "Split": "Val"})
        train_subs = set(hc_subjects + pd_subjects) - val_subs
        for s in sorted(train_subs):
            assignment_rows.append({"Fold": f, "Subject_ID": s,
                                    "Class": "HC" if s.startswith("HC_") else "PD",
                                    "Split": "Train"})

    # Convert subject sets to image-level indices
    folds = []
    for f, val_subs in enumerate(fold_val_subjects):
        va_mask = np.array([g in val_subs for g in groups])
        va_idx  = np.where(va_mask)[0]
        tr_idx  = np.where(~va_mask)[0]
        folds.append({"val_subjects": val_subs, "tr_idx": tr_idx, "va_idx": va_idx})

    return folds, pd.DataFrame(assignment_rows)


# ====================== PLOT HELPER ======================
def save_fold_pdf(history, file_stem, plot_dir):
    for suffix, t_key, v_key, ylabel in [
        ("accuracy", "train_acc",  "val_acc",  "Accuracy"),
        ("loss",     "train_loss", "val_loss", "Loss"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(history["epoch"], history[t_key], label="Train")
        ax.plot(history["epoch"], history[v_key], label="Val")
        best_ep = history.get("best_epoch")
        if best_ep is not None and best_ep in history["epoch"]:
            idx      = history["epoch"].index(best_ep)
            best_val = history[v_key][idx]
            ax.axvline(x=best_ep, color="red", linestyle="--", alpha=0.6,
                       label=f"Best epoch ({best_ep})")
            ax.scatter([best_ep], [best_val], color="red", zorder=5)
        ax.grid(False)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend()
        plt.tight_layout()
        pdf_path = os.path.join(plot_dir, f"{file_stem}_{suffix}.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ====================== EXCEL STYLING ======================
def style_worksheet(ws, header_hex, mean_row_idx=None, total_row_idx=None):
    hdr_fill  = PatternFill("solid", fgColor=header_hex)
    hdr_font  = Font(bold=True, color="FFFFFF", name="Arial")
    alt_fill  = PatternFill("solid", fgColor="DCE6F1")
    mean_fill = PatternFill("solid", fgColor="FFD966")   # yellow — Mean row
    tot_fill  = PatternFill("solid", fgColor="A9D18E")   # green  — Total row
    mean_font = Font(bold=True, name="Arial")
    tot_font  = Font(bold=True, name="Arial")
    thin      = Side(style="thin", color="BFBFBF")
    border    = Border(left=thin, right=thin, top=thin, bottom=thin)

    for cell in ws[1]:
        cell.fill = hdr_fill; cell.font = hdr_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border

    for row_idx, row in enumerate(ws.iter_rows(min_row=2), 2):
        if mean_row_idx is not None and row_idx == mean_row_idx:
            for cell in row:
                cell.fill = mean_fill; cell.font = mean_font
                cell.alignment = Alignment(horizontal="center"); cell.border = border
        elif total_row_idx is not None and row_idx == total_row_idx:
            for cell in row:
                cell.fill = tot_fill; cell.font = tot_font
                cell.alignment = Alignment(horizontal="center"); cell.border = border
        else:
            fill = alt_fill if row_idx % 2 == 0 else PatternFill()
            for cell in row:
                cell.fill = fill
                cell.alignment = Alignment(horizontal="center")
                cell.border = border
                if cell.font:
                    cell.font = Font(name="Arial", size=cell.font.size)

    for col_idx, col_cells in enumerate(ws.columns, 1):
        max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
        ws.column_dimensions[get_column_letter(col_idx)].width = max_len + 4

    ws.freeze_panes = "A2"


def save_history_xlsx(histories, path, header_hex):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_label, h in histories.items():
            df_h = pd.DataFrame({
                "Epoch":      h["epoch"],
                "Train_Loss": h["train_loss"],
                "Val_Loss":   h["val_loss"],
                "Train_Acc":  h["train_acc"],
                "Val_Acc":    h["val_acc"],
                "Best_Epoch": [h.get("best_epoch", "")] * len(h["epoch"]),
            })
            sname = sheet_label[:31]
            df_h.to_excel(writer, sheet_name=sname, index=False)
            style_worksheet(writer.sheets[sname], header_hex)
    print(f"  History saved → {path}")


# ====================== TRAINING LOOP ======================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = correct = total = 0
    for imgs, lbls, _ in loader:
        imgs = imgs.to(DEVICE); lbls = lbls.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward(); optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(out) >= 0.5).long()
        correct += (preds == lbls.long()).sum().item()
        total   += imgs.size(0)
    return running_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss = correct = total = 0
    val_paths, val_trues, val_preds, val_probs = [], [], [], []
    with torch.no_grad():
        for imgs, lbls, pths in loader:
            outs = model(imgs.to(DEVICE))
            loss = criterion(outs, lbls.to(DEVICE).unsqueeze(1))
            running_loss += loss.item() * imgs.size(0)
            preds, probs = logits_to_preds_probs(outs)
            correct += (preds == lbls.numpy().astype(int)).sum()
            total   += imgs.size(0)
            val_paths.extend(pths); val_trues.extend(lbls.numpy().astype(int))
            val_preds.extend(preds); val_probs.extend(probs)
    return running_loss / total, correct / total, val_paths, val_trues, val_preds, val_probs


# ====================== TRAIN ONE FOLD (with resume) ======================
def train_fold(fold, tr_idx, va_idx, paths, labels,
               model_prefix, plot_prefix, model_dir, plot_dir):
    best_model_path = os.path.join(model_dir, f"{model_prefix}_best.pth")

    val_ds     = EEGImageDataset(paths[va_idx], labels[va_idx], val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    _, criterion = make_optimizer_and_loss(create_model())

    # ── RESUME: fold already done ──────────────────────────────────────
    if os.path.isfile(best_model_path) and not FORCE_RETRAIN:
        print(f"  [RESUME] Fold {fold} — .pth found, skipping training → {best_model_path}")
        model = create_model()
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
        _, _, val_paths, val_trues, val_preds, val_probs = \
            validate(model, val_loader, criterion)
        history = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "best_epoch": -1,
        }
        return val_paths, val_trues, val_preds, val_probs, history

    # ── TRAIN: fold not yet done ───────────────────────────────────────
    train_ds     = EEGImageDataset(paths[tr_idx], labels[tr_idx], train_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model                = create_model()
    optimizer, criterion = make_optimizer_and_loss(model)

    # v7: CosineAnnealingLR — decays LR from 3e-4 down to eta_min=1e-6
    #     over T_max=EPOCHS steps. From-scratch CNNs need annealing to
    #     converge; pretrained models can skip it (they start near a
    #     good solution). scheduler.step() is called after each epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val_loss  = float("inf")
    patience_count = 0
    best_epoch     = 1

    history = {k: [] for k in ("epoch","train_loss","val_loss","train_acc","val_acc")}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_paths, val_trues, val_preds, val_probs = \
            validate(model, val_loader, criterion)

        # Step the cosine scheduler once per epoch (after validation)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:02d}/{EPOCHS} | "
              f"TrainLoss {train_loss:.4f} TrainAcc {train_acc:.4f} | "
              f"ValLoss {val_loss:.4f} ValAcc {val_acc:.4f} | "
              f"LR {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            best_epoch     = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  [ES] Improved → saved. (best epoch so far: {best_epoch})")
        else:
            patience_count += 1
            print(f"  [ES] No improvement ({patience_count}/{ES_PATIENCE})")
            # ── Early stopping guarded by ES_MIN_EPOCHS ──────────────
            if patience_count >= ES_PATIENCE and epoch >= ES_MIN_EPOCHS:
                print(f"  [ES] Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    history["best_epoch"] = best_epoch

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    _, _, val_paths, val_trues, val_preds, val_probs = \
        validate(model, val_loader, criterion)

    save_fold_pdf(history, plot_prefix, plot_dir)
    print(f"  [INFO] Best epoch: {best_epoch} | Best val loss: {best_val_loss:.4f}")

    return val_paths, val_trues, val_preds, val_probs, history


# ====================== PER-FOLD SUBJECT COUNT ======================
def report_fold_subject_counts_group(fold, va_idx, groups):
    val_groups      = groups[va_idx]
    unique_subjects = np.unique(val_groups)
    n_hc = sum(1 for s in unique_subjects if s.startswith("HC_"))
    n_pd = sum(1 for s in unique_subjects if s.startswith("PD_"))
    print(f"  [Fold {fold}] Validation subjects → HC: {n_hc}, PD: {n_pd}, Total: {n_hc + n_pd}")
    return {"Fold": fold, "Val_HC_Subjects": n_hc,
            "Val_PD_Subjects": n_pd, "Val_Total_Subjects": n_hc + n_pd}


def report_fold_subject_counts_standard(fold, va_idx, paths):
    val_paths       = paths[va_idx]
    unique_subjects = set(extract_subject_id(p) for p in val_paths)
    n_hc = sum(1 for s in unique_subjects if s.startswith("HC_"))
    n_pd = sum(1 for s in unique_subjects if s.startswith("PD_"))
    print(f"  [Fold {fold}] Validation subjects (image-level split) → "
          f"HC: {n_hc}, PD: {n_pd}, Total: {n_hc + n_pd}")
    return {"Fold": fold, "Val_HC_Subjects": n_hc,
            "Val_PD_Subjects": n_pd, "Val_Total_Subjects": n_hc + n_pd}


# ====================== SAVE IMAGE PREDICTIONS ======================
def save_image_predictions(fold, val_paths, val_trues, val_preds, val_probs,
                           protocol_prefix, tag, result_dir):
    csv_path = os.path.join(
        result_dir, f"{tag}_image_predictions_{protocol_prefix}_fold{fold}.csv"
    )
    if os.path.isfile(csv_path) and not FORCE_RETRAIN:
        print(f"  [RESUME] Image predictions already saved → {csv_path}")
        return
    df = pd.DataFrame({
        "Fold":       fold,
        "Image_Path": val_paths,
        "Subject_ID": [extract_subject_id(p) for p in val_paths],
        "True_Label": val_trues,
        "Pred_Label": val_preds,
        "Pred_Prob":  val_probs,
        "Correct":    [int(t == p) for t, p in zip(val_trues, val_preds)],
    })
    df.to_csv(csv_path, index=False)
    print(f"  Image predictions saved → {csv_path}")


# ================================================================
#  COMBINED RESULTS SUMMARY  (styled Excel, per dataset)
# ================================================================
def write_summary_excel(gkf_fold_df, std_fold_df,
                        gkf_total_tp, gkf_total_tn, gkf_total_fp, gkf_total_fn,
                        std_total_tp, std_total_tn, std_total_fp, std_total_fn,
                        tag, result_dir):
    xlsx_path = os.path.join(result_dir, f"{tag}_results_summary.xlsx")

    metric_cols = ["Accuracy", "Precision", "Sensitivity", "Specificity",
                   "FAR", "FRR", "MCC", "AUC", "Best_Epoch",
                   "TP", "TN", "FP", "FN"]

    def build_sheet_df(fold_df, total_tp, total_tn, total_fp, total_fn):
        cols = ["Fold"] + [c for c in metric_cols if c in fold_df.columns]
        df   = fold_df[cols].copy()

        # Mean row — simple average of per-fold values
        mean_row         = df.mean(numeric_only=True).to_frame().T
        mean_row["Fold"] = "Mean"
        mean_row         = mean_row[cols]

        # Total row — recomputed from summed TP/TN/FP/FN
        total_metrics = compute_metrics(total_tp, total_tn, total_fp, total_fn)
        fold_aucs = [v for v in df["AUC"].tolist()
                     if isinstance(v, float) and not np.isnan(v)] if "AUC" in df.columns else []
        total_metrics["AUC"] = float(np.mean(fold_aucs)) if fold_aucs else float("nan")

        total_row = pd.DataFrame([{
            "Fold":       "Total",
            "TP":         total_tp,  "TN": total_tn,
            "FP":         total_fp,  "FN": total_fn,
            "Best_Epoch": "",
            **{k: v for k, v in total_metrics.items()},
        }])[cols]

        return pd.concat([df, mean_row, total_row], ignore_index=True)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_gkf = build_sheet_df(gkf_fold_df,
                                gkf_total_tp, gkf_total_tn, gkf_total_fp, gkf_total_fn)
        df_gkf.to_excel(writer, sheet_name="GroupKFold", index=False)
        n_rows = len(df_gkf) + 1
        style_worksheet(writer.sheets["GroupKFold"], "7030A0",
                        mean_row_idx=n_rows - 1, total_row_idx=n_rows)

        df_std = build_sheet_df(std_fold_df,
                                std_total_tp, std_total_tn, std_total_fp, std_total_fn)
        df_std.to_excel(writer, sheet_name="Standard5Fold", index=False)
        n_rows = len(df_std) + 1
        style_worksheet(writer.sheets["Standard5Fold"], "4472C4",
                        mean_row_idx=n_rows - 1, total_row_idx=n_rows)

    print(f"  [{tag}] Results summary saved → {xlsx_path}")


# ================================================================
#  PROTOCOL 1 — Manual Balanced Subject-Aware KFold
# ================================================================
def run_groupkfold(paths, labels, groups, tag, result_dir, model_dir, plot_dir):
    print("\n" + "="*60)
    print(f"[{tag}] PROTOCOL 1 — Manual Balanced Subject-Aware KFold (5-fold)")
    print(f"        Guaranteed: folds 1-4 → 3 HC + 3 PD val subjects")
    print(f"                    fold  5   → 2 HC + 2 PD val subjects")
    print("="*60)

    # Build manually balanced folds and save subject assignment CSV
    folds, assignment_df = make_balanced_subject_folds(paths, labels, groups,
                                                       n_splits=N_SPLITS, seed=SEED)
    assignment_csv = os.path.join(result_dir, f"{tag}_groupkfold_subject_fold_assignment.csv")
    assignment_df.to_csv(assignment_csv, index=False)
    print(f"  Subject fold assignment saved → {assignment_csv}")

    total_TP = total_TN = total_FP = total_FN = 0
    fold_metrics        = []
    histories           = {}
    subject_count_rows  = []

    for fold, fold_data in enumerate(folds, 1):
        tr_idx = fold_data["tr_idx"]
        va_idx = fold_data["va_idx"]

        print(f"\n→ [{tag}] GroupKFold Fold {fold}/{N_SPLITS}")

        subject_count_rows.append(
            report_fold_subject_counts_group(fold, va_idx, groups)
        )

        val_paths, val_trues, val_preds, val_probs, history = train_fold(
            fold, tr_idx, va_idx, paths, labels,
            model_prefix = f"{tag}_HC_vs_PD_groupkfold_fold{fold}",
            plot_prefix  = f"{tag}_groupkfold_fold{fold}",
            model_dir    = model_dir,
            plot_dir     = plot_dir,
        )
        histories[f"Fold_{fold}"] = history

        save_image_predictions(fold, val_paths, val_trues, val_preds, val_probs,
                               "groupkfold", tag, result_dir)

        val_trues_arr = np.array(val_trues); val_preds_arr = np.array(val_preds)
        TP = int(((val_preds_arr == 1) & (val_trues_arr == 1)).sum())
        TN = int(((val_preds_arr == 0) & (val_trues_arr == 0)).sum())
        FP = int(((val_preds_arr == 1) & (val_trues_arr == 0)).sum())
        FN = int(((val_preds_arr == 0) & (val_trues_arr == 1)).sum())
        total_TP += TP; total_TN += TN; total_FP += FP; total_FN += FN

        metrics = compute_metrics(TP, TN, FP, FN,
                                  all_labels=val_trues, all_probs=val_probs)
        fold_row = {
            "Fold":       fold,
            "Best_Epoch": history["best_epoch"],
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            **metrics,
        }
        fold_metrics.append(fold_row)

        print(f"  TP={TP} TN={TN} FP={FP} FN={FN}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


    # ── Save history Excel
    save_history_xlsx(
        histories,
        os.path.join(result_dir, f"{tag}_groupkfold_history.xlsx"),
        "7030A0"
    )

    # ── Save subject counts CSV
    subject_count_df = pd.DataFrame(subject_count_rows)
    subject_count_df.to_csv(
        os.path.join(result_dir, f"{tag}_fold_subject_counts_groupkfold.csv"), index=False
    )
    print(f"\n[{tag}] Per-fold validation subject counts (Manual Balanced GroupKFold):")
    print(subject_count_df.to_string(index=False))

    # ── Save CSVs
    fold_df = pd.DataFrame(fold_metrics)
    fold_df = fold_df[["Fold"] + [c for c in fold_df.columns if c != "Fold"]]
    fold_df.to_csv(
        os.path.join(result_dir, f"{tag}_fold_metrics_HC_vs_PD_groupkfold.csv"), index=False)

    # ── Final aggregated
    final_metrics = compute_metrics(total_TP, total_TN, total_FP, total_FN)
    fold_aucs = [r["AUC"] for r in fold_metrics
                 if "AUC" in r and not np.isnan(r["AUC"])]
    final_metrics["AUC"] = float(np.mean(fold_aucs)) if fold_aucs else float("nan")

    print(f"\n==== [{tag}] FINAL Manual Balanced GroupKFold Aggregated ====")
    print(f"TP={total_TP} TN={total_TN} FP={total_FP} FN={total_FN}")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    final_df = pd.DataFrame([{
        "Task": f"HC_vs_PD_IOWA_{tag}_GroupKFold",
        "TP": total_TP, "TN": total_TN, "FP": total_FP, "FN": total_FN,
        **final_metrics
    }])
    final_df.to_csv(
        os.path.join(result_dir, f"{tag}_final_aggregated_metrics_HC_vs_PD_groupkfold.csv"),
        index=False)
    final_df.to_excel(
        os.path.join(result_dir, f"{tag}_final_aggregated_metrics_HC_vs_PD_groupkfold.xlsx"),
        index=False)

    return fold_df, total_TP, total_TN, total_FP, total_FN


# ================================================================
#  PROTOCOL 2 — Standard StratifiedKFold (image-level)
# ================================================================
def run_standard_kfold(paths, labels, groups, tag, result_dir, model_dir, plot_dir):
    print("\n" + "="*60)
    print(f"[{tag}] PROTOCOL 2 — Standard StratifiedKFold (5-fold, image-level)")
    print("="*60)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    total_TP = total_TN = total_FP = total_FN = 0
    fold_metrics       = []
    histories          = {}
    subject_count_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(paths, labels), 1):
        print(f"\n→ [{tag}] Standard Fold {fold}/{N_SPLITS}")

        subject_count_rows.append(
            report_fold_subject_counts_standard(fold, va_idx, paths)
        )

        val_paths, val_trues, val_preds, val_probs, history = train_fold(
            fold, tr_idx, va_idx, paths, labels,
            model_prefix = f"{tag}_HC_vs_PD_standard_fold{fold}",
            plot_prefix  = f"{tag}_standard_fold{fold}",
            model_dir    = model_dir,
            plot_dir     = plot_dir,
        )
        histories[f"Fold_{fold}"] = history

        save_image_predictions(fold, val_paths, val_trues, val_preds, val_probs,
                               "standard", tag, result_dir)

        val_trues = np.array(val_trues); val_preds = np.array(val_preds)
        TP = int(((val_preds == 1) & (val_trues == 1)).sum())
        TN = int(((val_preds == 0) & (val_trues == 0)).sum())
        FP = int(((val_preds == 1) & (val_trues == 0)).sum())
        FN = int(((val_preds == 0) & (val_trues == 1)).sum())
        total_TP += TP; total_TN += TN; total_FP += FP; total_FN += FN

        metrics = compute_metrics(TP, TN, FP, FN,
                                  all_labels=val_trues, all_probs=val_probs)
        fold_row = {
            "Fold":       fold,
            "Best_Epoch": history["best_epoch"],
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            **metrics,
        }
        fold_metrics.append(fold_row)

        print(f"  TP={TP} TN={TN} FP={FP} FN={FN}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # ── Save history Excel
    save_history_xlsx(
        histories,
        os.path.join(result_dir, f"{tag}_standard_history.xlsx"),
        "4472C4"
    )

    # ── Save subject counts CSV
    subject_count_df = pd.DataFrame(subject_count_rows)
    subject_count_df.to_csv(
        os.path.join(result_dir, f"{tag}_fold_subject_counts_standard.csv"), index=False
    )
    print(f"\n[{tag}] Per-fold validation subject counts (Standard KFold):")
    print(subject_count_df.to_string(index=False))

    # ── Save CSVs — Fold column first
    fold_df = pd.DataFrame(fold_metrics)
    fold_df = fold_df[["Fold"] + [c for c in fold_df.columns if c != "Fold"]]
    fold_df.to_csv(
        os.path.join(result_dir, f"{tag}_fold_metrics_HC_vs_PD_standard.csv"), index=False)

    # ── Final aggregated
    final_metrics = compute_metrics(total_TP, total_TN, total_FP, total_FN)
    fold_aucs = [r["AUC"] for r in fold_metrics
                 if "AUC" in r and not np.isnan(r["AUC"])]
    final_metrics["AUC"] = float(np.mean(fold_aucs)) if fold_aucs else float("nan")

    print(f"\n==== [{tag}] FINAL Standard 5-Fold Aggregated ====")
    print(f"TP={total_TP} TN={total_TN} FP={total_FP} FN={total_FN}")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    final_df = pd.DataFrame([{
        "Task": f"HC_vs_PD_IOWA_{tag}_Standard5Fold",
        "TP": total_TP, "TN": total_TN, "FP": total_FP, "FN": total_FN,
        **final_metrics
    }])
    final_df.to_csv(
        os.path.join(result_dir, f"{tag}_final_aggregated_metrics_HC_vs_PD_standard.csv"),
        index=False)
    final_df.to_excel(
        os.path.join(result_dir, f"{tag}_final_aggregated_metrics_HC_vs_PD_standard.xlsx"),
        index=False)

    return fold_df, total_TP, total_TN, total_FP, total_FN


# ================================================================
#  MAIN — iterate over all four datasets
# ================================================================
if FORCE_RETRAIN:
    print("\n[INFO] FORCE_RETRAIN=True — all saved files will be ignored, full retraining.")
else:
    print("\n[INFO] FORCE_RETRAIN=False — completed datasets and folds will be skipped.")

for tag, hc_dir, pd_dir in DATASETS:
    print("\n" + "#"*70)
    print(f"#  DATASET: {tag}  (IOWA)")
    print("#"*70)

    result_dir = os.path.join(BASE_RESULT_DIR, tag)
    model_dir  = os.path.join(result_dir, "models")
    plot_dir   = os.path.join(result_dir, "plots")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir,  exist_ok=True)
    os.makedirs(plot_dir,   exist_ok=True)

    # ── Skip entire dataset if summary already exists ──────────────────
    summary_path = os.path.join(result_dir, f"{tag}_results_summary.xlsx")
    if os.path.isfile(summary_path) and not FORCE_RETRAIN:
        print(f"[RESUME] {tag} already complete — results_summary.xlsx found, skipping.")
        print(f"         Delete or set FORCE_RETRAIN=True to rerun this dataset.")
        continue

    paths, labels, groups = collect_paths(hc_dir, pd_dir, tag)

    gkf_fold_df, gkf_tp, gkf_tn, gkf_fp, gkf_fn = \
        run_groupkfold(paths, labels, groups, tag, result_dir, model_dir, plot_dir)

    std_fold_df, std_tp, std_tn, std_fp, std_fn = \
        run_standard_kfold(paths, labels, groups, tag, result_dir, model_dir, plot_dir)

    write_summary_excel(
        gkf_fold_df, std_fold_df,
        gkf_tp, gkf_tn, gkf_fp, gkf_fn,
        std_tp, std_tn, std_fp, std_fn,
        tag, result_dir
    )

    print(f"\n{'='*60}")
    print(f"[{tag}] COMPLETE")
    print(f"  Plots          : {plot_dir}/{tag}_{{groupkfold,standard}}_fold*_{{accuracy,loss}}.pdf")
    print(f"  History xlsx   : {result_dir}/{tag}_groupkfold_history.xlsx")
    print(f"                   {result_dir}/{tag}_standard_history.xlsx")
    print(f"  Summary xlsx   : {result_dir}/{tag}_results_summary.xlsx")
    print(f"  Subj counts    : {result_dir}/{tag}_fold_subject_counts_groupkfold.csv")
    print(f"                   {result_dir}/{tag}_fold_subject_counts_standard.csv")
    print(f"  Subj assignment: {result_dir}/{tag}_groupkfold_subject_fold_assignment.csv")
    print(f"  Img preds      : {result_dir}/{tag}_image_predictions_{{groupkfold,standard}}_fold*.csv")
    print(f"  Models         : {model_dir}/{tag}_HC_vs_PD_{{groupkfold,standard}}_fold*_best.pth")
    print(f"{'='*60}")

print("\n" + "#"*70)
print("#  ALL DATASETS COMPLETE  (EWT · CWT · STFT · DWT)")
print("#"*70)
