"""
evaluate.py
-----------
Evaluates the global federated model on the full test set.

Produces:
  - Accuracy, F1, AUC-ROC, Sensitivity, Specificity per class
  - Confusion matrix plot
  - ROC curves per class

Run after training:
    python evaluate.py --checkpoint ./outputs/checkpoints/round_010.pt
                       --data_dir ./data
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report,
)
from sklearn.preprocessing import label_binarize

from dataset import GBCUDataset, get_transforms, CLASS_NAMES, IDX_TO_CLASS
from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./outputs"


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_probs):
    num_classes = len(CLASS_NAMES)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # AUC-ROC (one-vs-rest)
    try:
        auc_macro = roc_auc_score(y_bin, y_probs, multi_class="ovr", average="macro")
        auc_per_class = roc_auc_score(y_bin, y_probs, average=None)
    except ValueError:
        auc_macro = 0.0
        auc_per_class = [0.0] * num_classes

    # Per-class sensitivity (recall) and specificity
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sensitivity = []
    specificity = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity.append(sens)
        specificity.append(spec)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "auc_macro": auc_macro,
        "auc_per_class": auc_per_class,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm,
    }


def print_metrics(metrics: dict):
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS — Global Federated Model")
    print("=" * 55)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  AUC (macro): {metrics['auc_macro']:.4f}")
    print("-" * 55)
    print(f"  {'Class':<12} {'F1':>6} {'AUC':>6} {'Sens':>7} {'Spec':>7}")
    print("-" * 55)
    for i, cls in enumerate(CLASS_NAMES):
        print(
            f"  {cls:<12} "
            f"{metrics['f1_per_class'][i]:>6.4f} "
            f"{metrics['auc_per_class'][i]:>6.4f} "
            f"{metrics['sensitivity'][i]:>7.4f} "
            f"{metrics['specificity'][i]:>7.4f}"
        )
    print("=" * 55 + "\n")


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Global Federated Model")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Evaluate] Confusion matrix saved -> {save_path}")


def plot_roc_curves(y_true, y_probs, save_path: str):
    num_classes = len(CLASS_NAMES)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue", "darkorange", "green", "crimson", "purple"]

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, color=colors[i],
                label=f"{CLASS_NAMES[i]} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Global Federated Model")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Evaluate] ROC curves saved -> {save_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    print(f"[Evaluate] Loading checkpoint: {args.checkpoint}")
    print(f"[Evaluate] Device: {DEVICE}")

    model = build_model(pretrained=False).to(DEVICE)
    state_dict = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)

    test_dataset = GBCUDataset(args.data_dir, transform=get_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[Evaluate] Test samples: {len(test_dataset)}")

    y_true, y_pred, y_probs = run_inference(model, test_loader)
    metrics = compute_metrics(y_true, y_pred, y_probs)
    print_metrics(metrics)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
    plot_roc_curves(
        y_true, y_probs,
        save_path=os.path.join(OUTPUT_DIR, "roc_curves.png")
    )

    print("[Evaluate] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()
