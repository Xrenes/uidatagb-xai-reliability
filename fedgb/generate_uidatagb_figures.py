"""
generate_uidatagb_figures.py
------------------------------
Generates confusion matrices, ROC curves, per-class metrics chart, and
dataset statistics chart for the UIdataGB primary model (9 classes),
from real model inference -- replacing the stale 5-class GBCU figures
that were still sitting in outputs/figures/.

Run from fedgb/:  python generate_uidatagb_figures.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc as sklearn_auc
from sklearn.preprocessing import label_binarize
import torch

from dataset import GBCUDataset, get_transforms, CLASS_NAMES, CLASS_LABELS, CLASS_SHORT
from model import build_model
from evaluate import run_inference, compute_metrics, DEVICE

HERE = os.path.dirname(os.path.abspath(__file__))
# Stage-1 defaults, unchanged behaviour; set P0_* env vars to point at the
# Stage-2 (batch-corrected) checkpoint/data/output dirs instead (see
# run_stage2_pipeline.py), without touching this file per invocation.
OUTPUT_DIR = os.environ.get("P0_FIG_DIR", os.path.join(HERE, "outputs", "figures"))
CKPT = os.environ.get("P0_CKPT", "./outputs/seeds/seed_42/centralized.pt")
DATA_DIR = os.environ.get("P0_DATA_DIR", "./data/uidatagb")
PHASE0_DIR = os.environ.get("P0_OUT_DIR", os.path.join(HERE, "outputs", "phase0"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PHASE0_DIR, exist_ok=True)
N = len(CLASS_NAMES)
SHORT_NAMES = [CLASS_SHORT[c] for c in CLASS_NAMES]
COLORS = plt.cm.tab10(np.linspace(0, 1, N))


def main():
    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

    val_dataset = GBCUDataset(os.path.join(DATA_DIR, "validation"), transform=get_transforms(train=False))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"[figs] loaded {len(val_dataset)} validation images")

    y_true, y_pred, y_probs = run_inference(model, val_loader)
    metrics = compute_metrics(y_true, y_pred, y_probs)
    cm = metrics["confusion_matrix"]
    print(f"[figs] accuracy={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} auc={metrics['auc_macro']:.4f}")

    # ---- Normalized confusion matrix ----
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(SHORT_NAMES, fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels(SHORT_NAMES, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    thresh = cm_norm.max() / 2.0
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm_norm[i, j] > thresh else "black")
    ax.set_title("Normalized Confusion Matrix – Primary Model\n"
                "(Row-normalized; diagonal = per-class recall; real predictions)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_normalized.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # ---- Absolute confusion matrix ----
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(SHORT_NAMES, fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels(SHORT_NAMES, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    thresh = cm.max() / 2.0
    for i in range(N):
        for j in range(N):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_title(f"Absolute Confusion Matrix – Primary Model\n"
                f"(Counts on {len(val_dataset)}-sample validation set; real predictions)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_absolute.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # ---- ROC curves ----
    # Single panel, zoomed directly into the diagnostic region (FPR 0-0.2,
    # TPR 0.8-1.0) rather than the full 0-1 square: with every class AUC
    # >= 0.99, the curves are indistinguishable at full scale, and this
    # range is where the per-class separation actually shows up.
    y_bin = label_binarize(y_true, classes=list(range(N)))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for i in range(N):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], lw=1.8, label=f"{SHORT_NAMES[i]} (AUC={roc_auc:.3f})")
    ax.plot([0, 0.2], [0.8, 1.0], "k--", lw=1, label="Random (AUC=0.50)")
    ax.set_xlim(0.0, 0.2)
    ax.set_ylim(0.8, 1.0)
    ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"Multi-Class ROC Curves – Primary Model\n"
                f"(One-vs-Rest, {N}-class gallbladder disease classification)")
    ax.legend(loc="lower right", fontsize=7.5, ncol=1)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # ---- Per-class metrics bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N)
    width = 0.2
    ax.bar(x - 1.5*width, metrics["f1_per_class"], width, label="F1", color="#1768c4")
    ax.bar(x - 0.5*width, metrics["auc_per_class"], width, label="AUC-ROC", color="#0f6e5c")
    ax.bar(x + 0.5*width, metrics["sensitivity"], width, label="Sensitivity", color="#d9601a")
    ax.bar(x + 1.5*width, metrics["specificity"], width, label="Specificity", color="#6a2fa6")
    ax.set_xticks(x); ax.set_xticklabels(SHORT_NAMES, fontsize=8, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class F1, AUC-ROC, Sensitivity, and Specificity – Primary Model")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "per_class_metrics.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # ---- Dataset statistics chart ----
    train_ds = GBCUDataset(os.path.join(DATA_DIR, "training"), transform=get_transforms(train=False))
    train_counts = [0] * N
    val_counts = [0] * N
    for _, l in train_ds.samples:
        train_counts[l] += 1
    for _, l in val_dataset.samples:
        val_counts[l] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N)
    width = 0.35
    ax.bar(x - width/2, train_counts, width, label="Train", color="#1768c4")
    ax.bar(x + width/2, val_counts, width, label="Validation", color="#d9601a")
    ax.set_xticks(x); ax.set_xticklabels(SHORT_NAMES, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Per-Class Training and Validation Sample Counts (UIdataGB)")
    ax.legend(fontsize=9)
    for i, (tr, va) in enumerate(zip(train_counts, val_counts)):
        ax.text(i - width/2, tr + 10, str(tr), ha="center", fontsize=7)
        ax.text(i + width/2, va + 10, str(va), ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dataset_statistics.png"), dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[figs] per-class train counts: {dict(zip(SHORT_NAMES, train_counts))}")
    print(f"[figs] per-class val counts:   {dict(zip(SHORT_NAMES, val_counts))}")
    print("[figs] wrote confusion_normalized.png, confusion_absolute.png, roc_curves.png, "
          "per_class_metrics.png, dataset_statistics.png")

    # also dump per-class metrics for the paper text
    import json
    per_class = {
        SHORT_NAMES[i]: {
            "label": CLASS_LABELS[CLASS_NAMES[i]],
            "f1": float(metrics["f1_per_class"][i]),
            "auc": float(metrics["auc_per_class"][i]),
            "sensitivity": float(metrics["sensitivity"][i]),
            "specificity": float(metrics["specificity"][i]),
            "train_n": train_counts[i],
            "val_n": val_counts[i],
        } for i in range(N)
    }
    out = {
        "accuracy": float(metrics["accuracy"]),
        "f1_macro": float(metrics["f1_macro"]),
        "auc_macro": float(metrics["auc_macro"]),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    os.makedirs(PHASE0_DIR, exist_ok=True)
    with open(os.path.join(PHASE0_DIR, "classification_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[figs] wrote outputs/phase0/classification_summary.json")


if __name__ == "__main__":
    main()
