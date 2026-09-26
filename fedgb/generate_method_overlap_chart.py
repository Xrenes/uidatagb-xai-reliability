"""
generate_method_overlap_chart.py
---------------------------------
Grouped bar chart of Table VI: cosine-overlap macro mean O_k^c per class,
compared across all five XAI methods (Grad-CAM, Grad-CAM++, Saliency,
Eigen-CAM, Score-CAM). Reads outputs/overlap_all_methods.json (already
computed for all five methods) and renders a single summary figure so
the numeric table also has a visual companion in the paper.

Run from fedgb/:
    python generate_method_overlap_chart.py
Output:
    outputs/figures/xai_method_overlap_comparison.png
"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(REPO_ROOT, "outputs", "overlap_all_methods.json")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "figures")

CLASS_ORDER = ["nml", "bmt", "stn", "abn", "malg"]
CLASS_LABELS = {
    "nml": "Normal", "bmt": "BMT", "stn": "Stones",
    "abn": "Abnormal", "malg": "Malignant",
}
METHOD_ORDER = ["gradcam", "gradcam_pp", "saliency", "eigencam", "scorecam"]
METHOD_LABELS = {
    "gradcam": "Grad-CAM", "gradcam_pp": "Grad-CAM++", "saliency": "Saliency Map",
    "eigencam": "Eigen-CAM", "scorecam": "Score-CAM",
}
COLORS = {
    "gradcam": "#4C72B0", "gradcam_pp": "#DD8452", "saliency": "#55A868",
    "eigencam": "#C44E52", "scorecam": "#8172B2",
}


def macro_mean(d, method, cls):
    vals = [d[method][cls][f"client_{k}"]["mean"] for k in range(3)]
    return float(np.mean(vals))


def main():
    with open(RESULTS_PATH) as f:
        d = json.load(f)

    x = np.arange(len(CLASS_ORDER))
    width = 0.16

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, method in enumerate(METHOD_ORDER):
        vals = [macro_mean(d, method, cls) for cls in CLASS_ORDER]
        offset = (i - (len(METHOD_ORDER) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=METHOD_LABELS[method],
                       color=COLORS[method], edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.axhline(0.70, color="#888888", linestyle="--", linewidth=1, alpha=0.8)
    ax.text(x[-1] + 2.2 * width, 0.70, r"$\theta$=0.70", fontsize=8.5,
             va="center", ha="left", color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER])
    ax.set_ylabel(r"Cosine-Overlap Macro Mean  $\bar{O}^c$")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Local-vs-Global Heatmap Agreement Across All Five XAI Methods\n"
        "(GBCU validation set, n=689, seed=42; averaged over $H_A, H_B, H_C$)"
    )
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "xai_method_overlap_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Figure] Saved -> {path}")


if __name__ == "__main__":
    main()
