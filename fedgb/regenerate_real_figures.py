"""
regenerate_real_figures.py
---------------------------
Regenerates confusion_normalized.png, confusion_absolute.png, and
roc_curves.png in outputs/figures/ from REAL model inference on the
real validation set, replacing the synthetic/reconstructed versions
previously produced by figures.py (which fabricated off-diagonal
confusion-matrix cells and approximated ROC curve shapes from AUC
values alone, rather than computing them from actual predictions).

Run from fedgb/:
    python regenerate_real_figures.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc as sklearn_auc
from sklearn.preprocessing import label_binarize

from dataset import GBCUDataset, get_transforms, CLASS_NAMES
from model import build_model
from evaluate import run_inference, compute_metrics, DEVICE

import torch

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
LONG_NAMES = ["Normal", "Benign\n(BMT)", "Stones", "Abnormal", "Malignant"]
COLORS = ["steelblue", "darkorange", "green", "crimson", "purple"]


def main():
    checkpoint = "./outputs/checkpoints/round_010.pt"
    data_dir = "./data/data/validation"

    model = build_model(pretrained=False).to(DEVICE)
    state_dict = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)

    val_dataset = GBCUDataset(data_dir, transform=get_transforms(train=False))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print(f"[Regen] Loaded {len(val_dataset)} real validation images")

    y_true, y_pred, y_probs = run_inference(model, val_loader)
    metrics = compute_metrics(y_true, y_pred, y_probs)
    cm = metrics["confusion_matrix"]
    print("[Regen] Real confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(f"[Regen] Accuracy={metrics['accuracy']:.4f}  F1={metrics['f1_macro']:.4f}  AUC={metrics['auc_macro']:.4f}")

    # ── Normalized confusion matrix ──────────────────────────────────────
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(LONG_NAMES, fontsize=9)
    ax.set_yticklabels(LONG_NAMES, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    thresh = cm_norm.max() / 2.0
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                     fontsize=10, color="white" if cm_norm[i, j] > thresh else "black")
    ax.set_title("Normalized Confusion Matrix - Primary Model\n"
                 "(Row-normalized; diagonal = per-class recall; real model predictions)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_normalized.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Regen] Saved -> {path}")

    # ── Absolute confusion matrix ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(LONG_NAMES, fontsize=9)
    ax.set_yticklabels(LONG_NAMES, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=10, color="white" if cm[i, j] > thresh else "black")
    ax.set_title("Absolute Confusion Matrix - Primary Model\n"
                 "(Counts on 689-sample validation set; real model predictions)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_absolute.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Regen] Saved -> {path}")

    # ── Real ROC curves ───────────────────────────────────────────────────
    y_bin = label_binarize(y_true, classes=list(range(5)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, col) in enumerate(zip(LONG_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name.replace(chr(10), ' ')} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Multi-Class ROC Curves - Primary Model\n"
                 "(One-vs-Rest, 5-class gallbladder disease classification; real predictions)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Regen] Saved -> {path}")

    print("\n[Regen] DONE. All three figures regenerated from real model inference.")


if __name__ == "__main__":
    main()
