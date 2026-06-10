# =========================================================================
# ✅ HC vs PD (IOWA) — EfficientNet-B0 (Pretrained, Last Layer Only)  [v1]
#    Datasets: EWT, CWT, STFT, DWT
#
#  TRAINING REGIME
#  ─────────────────────────────────────────────────────────────────────
#  - Pretrained ImageNet weights loaded for the full backbone.
#  - ALL backbone parameters are frozen (requires_grad=False).
#  - Only the final classification head is trainable.
#  - Effectively: L2-regularized logistic regression in ImageNet
#    feature space — converges rapidly, no LR scheduler needed.
#
#  DESIGN DECISIONS vs CUSTOM CNN (v7)  [for manuscript fairness]
#  ─────────────────────────────────────────────────────────────────────
#  IDENTICAL across all four models:
#    EPOCHS=20, BATCH_SIZE=32, ES_PATIENCE=3, ES_MIN_EPOCHS=7,
#    Adam optimizer, BCEWithLogitsLoss, weight_decay=1e-4,
#    fold splits (same seed + protocol), evaluation metrics,
#    subject-aware fold assignment.
#
#  LEGITIMATELY DIFFERENT (pretrained vs from-scratch):
#    LR = 1e-4  (vs 3e-4 for custom CNN)
#      — Single linear layer is well-conditioned; 1e-4 is standard
#        for last-layer-only fine-tuning.
#    No CosineAnnealingLR scheduler
#      — Backbone is frozen; only one linear layer is optimized.
#        That problem is approximately convex and converges in a
#        few epochs regardless of annealing. Scheduler omission is
#        standard practice in transfer learning literature.
#    RGB input + ImageNet normalization
#      — Required for pretrained backbone compatibility.
#    No Dropout2d in backbone (frozen; not updated).
#
#  TWO EVALUATION PROTOCOLS
#  ─────────────────────────────────────────────────────────────────────
#  1. Manual Balanced Subject-Aware KFold (5-fold)
#       Folds 1-4 : 3 HC + 3 PD validation subjects
#       Fold  5   : 2 HC + 2 PD validation subjects
#  2. Standard StratifiedKFold (5-fold, image-level)
#
#  RESUME / SKIP LOGIC
#  ─────────────────────────────────────────────────────────────────────
#  - results_summary.xlsx exists → dataset skipped entirely.
#  - fold .pth exists → training skipped, weights loaded for eval.
#  - FORCE_RETRAIN = True → ignore all saved files, retrain fully.
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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

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

BASE_RESULT_DIR = "/scratch/user/u.aa346327/results/efficientnet_b0"

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
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 1e-4      # Standard for last-layer-only fine-tuning
N_SPLITS    = 5
NUM_WORKERS = 8

ES_PATIENCE           = 3
ES_MIN_EPOCHS         = 7



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
# RGB input required for ImageNet-pretrained backbone.
# ImageNet normalization stats — mandatory for pretrained feature compatibility.
# RandomHorizontalFlip omitted — time-axis reversal is semantically invalid for EEG.
# RandomRotation omitted — breaks time-frequency axis meaning.
# RandomErasing: SpecAugment-style time/freq masking proxy (tensor-level, correct placement).
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.12, contrast=0.12),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.0))
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
        img = Image.open(self.paths[idx]).convert("RGB")   # RGB for pretrained backbone
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32), self.paths[idx]


# ====================== MODEL ======================
# EfficientNet-B0: freeze entire backbone, replace classifier head with Linear(1280→1).
# Only the new head parameters are updated during training.
def create_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features   # 1280
    model.classifier[1] = nn.Linear(in_features, 1) # newly created → requires_grad=True
    return model.to(DEVICE)


# Only trainable parameters (the new head) are passed to the optimizer.
# weight_decay=1e-4 consistent with custom CNN for manuscript comparability.
# No scheduler: single linear layer is approximately convex; converges
# rapidly without annealing. Standard for last-layer-only fine-tuning.
def make_optimizer_and_loss(model, lr=LR):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer        = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    criterion        = nn.BCEWithLogitsLoss()
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


# ====================== MANUAL BALANCED SUBJECT FOLD ASSIGNMENT ======================
def make_balanced_subject_folds(paths, labels, groups, n_splits=N_SPLITS, seed=SEED):
    rng = np.random.default_rng(seed)
    hc_subjects = sorted([s for s in np.unique(groups) if s.startswith("HC_")])
    pd_subjects = sorted([s for s in np.unique(groups) if s.startswith("PD_")])
    hc_subjects = list(rng.permutation(hc_subjects))
    pd_subjects = list(rng.permutation(pd_subjects))
    n_hc = len(hc_subjects)
    n_pd = len(pd_subjects)
    hc_fold_ids = [i % n_splits for i in range(n_hc)]
    pd_fold_ids = [i % n_splits for i in range(n_pd)]

    fold_val_subjects = []
    for f in range(n_splits):
        val_hc = {hc_subjects[i] for i, fid in enumerate(hc_fold_ids) if fid == f}
        val_pd = {pd_subjects[i] for i, fid in enumerate(pd_fold_ids) if fid == f}
        fold_val_subjects.append(val_hc | val_pd)

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
    mean_fill = PatternFill("solid", fgColor="FFD966")
    tot_fill  = PatternFill("solid", fgColor="A9D18E")
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


# ====================== TRAIN ONE FOLD ======================
def train_fold(fold, tr_idx, va_idx, paths, labels,
               model_prefix, plot_prefix, model_dir, plot_dir):
    best_model_path = os.path.join(model_dir, f"{model_prefix}_best.pth")

    val_ds     = EEGImageDataset(paths[va_idx], labels[va_idx], val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    _, criterion = make_optimizer_and_loss(create_model())

    if os.path.isfile(best_model_path) and not FORCE_RETRAIN:
        print(f"  [RESUME] Fold {fold} — .pth found, skipping training → {best_model_path}")
        model = create_model()
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
        _, _, val_paths, val_trues, val_preds, val_probs = \
            validate(model, val_loader, criterion)
        history = {"epoch": [], "train_loss": [], "val_loss": [],
                   "train_acc": [], "val_acc": [], "best_epoch": -1}
        return val_paths, val_trues, val_preds, val_probs, history

    train_ds     = EEGImageDataset(paths[tr_idx], labels[tr_idx], train_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model                = create_model()
    optimizer, criterion = make_optimizer_and_loss(model)

    best_val_loss  = float("inf")
    patience_count = 0
    best_epoch     = 1
    history        = {k: [] for k in ("epoch","train_loss","val_loss","train_acc","val_acc")}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_paths, val_trues, val_preds, val_probs = \
            validate(model, val_loader, criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:02d}/{EPOCHS} | "
              f"TrainLoss {train_loss:.4f} TrainAcc {train_acc:.4f} | "
              f"ValLoss {val_loss:.4f} ValAcc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            best_epoch     = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  [ES] Improved → saved. (best epoch so far: {best_epoch})")
        else:
            patience_count += 1
            print(f"  [ES] No improvement ({patience_count}/{ES_PATIENCE})")
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


# ====================== SUMMARY EXCEL ======================
def write_summary_excel(gkf_fold_df, std_fold_df,
                        gkf_total_tp, gkf_total_tn, gkf_total_fp, gkf_total_fn,
                        std_total_tp, std_total_tn, std_total_fp, std_total_fn,
                        tag, result_dir):
    xlsx_path   = os.path.join(result_dir, f"{tag}_results_summary.xlsx")
    metric_cols = ["Accuracy", "Precision", "Sensitivity", "Specificity",
                   "FAR", "FRR", "MCC", "AUC", "Best_Epoch", "TP", "TN", "FP", "FN"]

    def build_sheet_df(fold_df, total_tp, total_tn, total_fp, total_fn):
        cols      = ["Fold"] + [c for c in metric_cols if c in fold_df.columns]
        df        = fold_df[cols].copy()
        mean_row  = df.mean(numeric_only=True).to_frame().T
        mean_row["Fold"] = "Mean"
        mean_row  = mean_row[cols]
        total_metrics = compute_metrics(total_tp, total_tn, total_fp, total_fn)
        fold_aucs = [v for v in df["AUC"].tolist()
                     if isinstance(v, float) and not np.isnan(v)] if "AUC" in df.columns else []
        total_metrics["AUC"] = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
        total_row = pd.DataFrame([{
            "Fold": "Total", "TP": total_tp, "TN": total_tn,
            "FP": total_fp, "FN": total_fn, "Best_Epoch": "",
            **{k: v for k, v in total_metrics.items()},
        }])[cols]
        return pd.concat([df, mean_row, total_row], ignore_index=True)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_gkf = build_sheet_df(gkf_fold_df,
                                gkf_total_tp, gkf_total_tn, gkf_total_fp, gkf_total_fn)
        df_gkf.to_excel(writer, sheet_name="GroupKFold", index=False)
        n_rows = len(df_gkf) + 1
        style_worksheet(writer.sheets["GroupKFold"], "C55A11",
                        mean_row_idx=n_rows - 1, total_row_idx=n_rows)

        df_std = build_sheet_df(std_fold_df,
                                std_total_tp, std_total_tn, std_total_fp, std_total_fn)
        df_std.to_excel(writer, sheet_name="Standard5Fold", index=False)
        n_rows = len(df_std) + 1
        style_worksheet(writer.sheets["Standard5Fold"], "375623",
                        mean_row_idx=n_rows - 1, total_row_idx=n_rows)

    print(f"  [{tag}] Results summary saved → {xlsx_path}")


# ====================== PROTOCOL 1 — GroupKFold ======================
def run_groupkfold(paths, labels, groups, tag, result_dir, model_dir, plot_dir):
    print("\n" + "="*60)
    print(f"[{tag}] PROTOCOL 1 — Manual Balanced Subject-Aware KFold (5-fold)")
    print("="*60)

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
        subject_count_rows.append(report_fold_subject_counts_group(fold, va_idx, groups))

        val_paths, val_trues, val_preds, val_probs, history = train_fold(
            fold, tr_idx, va_idx, paths, labels,
            model_prefix = f"{tag}_HC_vs_PD_groupkfold_fold{fold}",
            plot_prefix  = f"{tag}_groupkfold_fold{fold}",
            model_dir    = model_dir, plot_dir = plot_dir,
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

        metrics  = compute_metrics(TP, TN, FP, FN, all_labels=val_trues, all_probs=val_probs)
        fold_row = {"Fold": fold, "Best_Epoch": history["best_epoch"],
                    "TP": TP, "TN": TN, "FP": FP, "FN": FN, **metrics}
        fold_metrics.append(fold_row)

        print(f"  TP={TP} TN={TN} FP={FP} FN={FN}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


    save_history_xlsx(histories,
                      os.path.join(result_dir, f"{tag}_groupkfold_history.xlsx"), "C55A11")

    subject_count_df = pd.DataFrame(subject_count_rows)
    subject_count_df.to_csv(
        os.path.join(result_dir, f"{tag}_fold_subject_counts_groupkfold.csv"), index=False)
    print(f"\n[{tag}] Per-fold validation subject counts (Manual Balanced GroupKFold):")
    print(subject_count_df.to_string(index=False))

    fold_df = pd.DataFrame(fold_metrics)
    fold_df = fold_df[["Fold"] + [c for c in fold_df.columns if c != "Fold"]]
    fold_df.to_csv(
        os.path.join(result_dir, f"{tag}_fold_metrics_HC_vs_PD_groupkfold.csv"), index=False)

    final_metrics     = compute_metrics(total_TP, total_TN, total_FP, total_FN)
    fold_aucs         = [r["AUC"] for r in fold_metrics
                         if "AUC" in r and not np.isnan(r["AUC"])]
    final_metrics["AUC"] = float(np.mean(fold_aucs)) if fold_aucs else float("nan")

    print(f"\n==== [{tag}] FINAL Manual Balanced GroupKFold Aggregated ====")
    print(f"TP={total_TP} TN={total_TN} FP={total_FP} FN={total_FN}")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    final_df = pd.DataFrame([{"Task": f"HC_vs_PD_IOWA_{tag}_GroupKFold",
                               "TP": total_TP, "TN": total_TN,
                               "FP": total_FP, "FN": total_FN, **final_metrics}])
    final_df.to_csv(
        os.path.join(result_dir, f"{tag}_final_aggregated_metrics_HC_vs_PD_groupkfold.csv"),
        index=False)
    final_df.to_excel(
        os.path.join(result_dir, f"{tag}_final_aggregated_metrics_HC_vs_PD_groupkfold.xlsx"),
        index=False)

    return fold_df, total_TP, total_TN, total_FP, total_FN


# ====================== PROTOCOL 2 — Standard StratifiedKFold ======================
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
        subject_count_rows.append(report_fold_subject_counts_standard(fold, va_idx, paths))

        val_paths, val_trues, val_preds, val_probs, history = train_fold(
            fold, tr_idx, va_idx, paths, labels,
            model_prefix = f"{tag}_HC_vs_PD_standard_fold{fold}",
            plot_prefix  = f"{tag}_standard_fold{fold}",
            model_dir    = model_dir, plot_dir = plot_dir,
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

        metrics  = compute_metrics(TP, TN, FP, FN, all_labels=val_trues, all_probs=val_probs)
        fold_row = {"Fold": fold, "Best_Epoch": history["best_epoch"],
                    "TP": TP, "TN": TN, "FP": FP, "FN": FN, **metrics}
        fold_metrics.append(fold_row)

        print(f"  TP={TP} TN={TN} FP={FP} FN={FN}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    save_history_xlsx(histories,
                      os.path.join(result_dir, f"{tag}_standard_history.xlsx"), "375623")

    subject_count_df = pd.DataFrame(subject_count_rows)
    subject_count_df.to_csv(
        os.path.join(result_dir, f"{tag}_fold_subject_counts_standard.csv"), index=False)
    print(f"\n[{tag}] Per-fold validation subject counts (Standard KFold):")
    print(subject_count_df.to_string(index=False))

    fold_df = pd.DataFrame(fold_metrics)
    fold_df = fold_df[["Fold"] + [c for c in fold_df.columns if c != "Fold"]]
    fold_df.to_csv(
        os.path.join(result_dir, f"{tag}_fold_metrics_HC_vs_PD_standard.csv"), index=False)

    final_metrics        = compute_metrics(total_TP, total_TN, total_FP, total_FN)
    fold_aucs            = [r["AUC"] for r in fold_metrics
                            if "AUC" in r and not np.isnan(r["AUC"])]
    final_metrics["AUC"] = float(np.mean(fold_aucs)) if fold_aucs else float("nan")

    print(f"\n==== [{tag}] FINAL Standard 5-Fold Aggregated ====")
    print(f"TP={total_TP} TN={total_TN} FP={total_FP} FN={total_FN}")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    final_df = pd.DataFrame([{"Task": f"HC_vs_PD_IOWA_{tag}_Standard5Fold",
                               "TP": total_TP, "TN": total_TN,
                               "FP": total_FP, "FN": total_FN, **final_metrics}])
    final_df.to_csv(
        os.path.join(result_dir, f"{tag}_final_aggregated_metrics_HC_vs_PD_standard.csv"),
        index=False)
    final_df.to_excel(
        os.path.join(result_dir, f"{tag}_final_aggregated_metrics_HC_vs_PD_standard.xlsx"),
        index=False)

    return fold_df, total_TP, total_TN, total_FP, total_FN


# ====================== MAIN ======================
if FORCE_RETRAIN:
    print("\n[INFO] FORCE_RETRAIN=True — full retraining.")
else:
    print("\n[INFO] FORCE_RETRAIN=False — completed datasets/folds will be skipped.")

for tag, hc_dir, pd_dir in DATASETS:
    print("\n" + "#"*70)
    print(f"#  DATASET: {tag}  (IOWA)  [EfficientNet-B0]")
    print("#"*70)

    result_dir = os.path.join(BASE_RESULT_DIR, tag)
    model_dir  = os.path.join(result_dir, "models")
    plot_dir   = os.path.join(result_dir, "plots")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir,  exist_ok=True)
    os.makedirs(plot_dir,   exist_ok=True)

    summary_path = os.path.join(result_dir, f"{tag}_results_summary.xlsx")
    if os.path.isfile(summary_path) and not FORCE_RETRAIN:
        print(f"[RESUME] {tag} already complete — skipping.")
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

print("\n" + "#"*70)
print("#  ALL DATASETS COMPLETE  [EfficientNet-B0]  (EWT · CWT · STFT · DWT)")
print("#"*70)
