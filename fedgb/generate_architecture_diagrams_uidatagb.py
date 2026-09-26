"""
generate_architecture_diagrams_uidatagb.py
--------------------------------------------
Regenerates the two pipeline/architecture diagrams for the UIdataGB paper
(Fig. 1a, 1b), which were previously stale GBCU-era images baked with the
wrong dataset size, split, and 5-class list.

Outputs:
  outputs/figures/xai_pipeline_diagram.png   (Fig. 1a)
  outputs/figures/resnet_backbone.png        (Fig. 1b)

Run from fedgb/:  python generate_architecture_diagrams_uidatagb.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.patheffects import withStroke

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SHORT_CODES = ["GS", "ART", "CHOL", "MGC", "PERF", "PCC", "ADNM", "CARC", "WT"]


def _box(ax, cx, cy, w, h, fc, title, subtitle, ec="#1a1a1a", lw=1.4,
         title_size=15, sub_size=11.5, text_color="white"):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.02",
                          linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)
    if subtitle:
        ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
                 fontsize=title_size, fontweight="bold", color=text_color, zorder=3)
        ax.text(cx, cy - h * 0.22, subtitle, ha="center", va="center",
                 fontsize=sub_size, style="italic", color=text_color, zorder=3,
                 linespacing=1.5)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                 fontsize=title_size, fontweight="bold", color=text_color, zorder=3)
    return box


def _arrow(ax, x1, y1, x2, y2, color="#424242", lw=1.6):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                           linewidth=lw, color=color, zorder=1)
    ax.add_patch(arr)


def fig_xai_pipeline():
    fig, ax = plt.subplots(figsize=(9.5, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15.6)
    ax.axis("off")

    ax.text(0.3, 15.2, "(a)  Classification and XAI Pipeline",
             fontsize=19, fontweight="bold", ha="left", va="top")

    w = 8.6
    cx = 5.0

    _box(ax, cx, 14.0, w, 1.15, "#111c2e",
         "UIdataGB Dataset (Site-Unlabelled Release)",
         "10,692 raw images, 9 classes\n"
         "leakage-corrected: train 7,499 / held-out val 3,193")

    _arrow(ax, cx, 13.42, cx, 13.02)
    _box(ax, cx, 12.55, w, 0.95, "#0d3f7a",
         "ResNet-50 Classifier",
         "ImageNet-pretrained backbone,\nfine-tuned end-to-end on the training set")

    _arrow(ax, cx, 12.07, cx, 11.67)
    _box(ax, cx, 11.15, w, 0.95, "#0f5fa8",
         "9-Class Prediction",
         "  ·  ".join(SHORT_CODES))

    _arrow(ax, cx, 10.67, cx, 10.27)
    _box(ax, cx, 9.75, w, 1.05, "#1a8fd1",
         "XAI Module  (layer4 activations)",
         "Grad-CAM  ·  Grad-CAM++  ·  Saliency  ·\nEigen-CAM  ·  Score-CAM")

    _arrow(ax, cx - 1.55, 9.22, cx - 2.2, 8.55)
    _arrow(ax, cx + 1.55, 9.22, cx + 2.2, 8.55)

    _box(ax, cx - 2.2, 7.75, 3.9, 1.75, "#0e7a5f",
         "Failure-Case\nGrounding",
         "confusion matrix → highest-\nconfidence misclassifications\ninspected via heatmap",
         title_size=14.5, sub_size=10.5)
    _box(ax, cx + 2.2, 7.75, 3.9, 1.75, "#c1560c",
         "Cross-Replicate\nConsistency",
         "3 independently trained model\nreplicates → cosine overlap $O^c$,\nflag threshold θ = 0.70",
         title_size=14.5, sub_size=10.5)

    _arrow(ax, cx - 2.2, 6.87, cx - 0.55, 6.55)
    _arrow(ax, cx + 2.2, 6.87, cx + 0.55, 6.55)

    _box(ax, cx, 5.95, w, 1.2, "#5c2a9d",
         "Quantitative XAI Evaluation",
         "faithfulness  ·  stability  ·  sanity check  ·\ncalibration  ·  shortcut audit")

    _arrow(ax, cx, 5.35, cx, 4.95)
    _box(ax, cx, 4.05, w, 1.55, "#1a7a3c",
         "Next-Stage Validation (this paper's roadmap)",
         "clinician ROI localization (IoU/Dice)  ·\npatient-level split verification  ·\nradiologist Likert study",
         title_size=14.5, sub_size=11)

    ax.text(cx, 2.9,
            "Single dataset release · site/scanner provenance unlabelled · internal validation only\n"
            "— not an externally validated multicentre or deployment claim",
            fontsize=11.5, style="italic", ha="center", va="top", color="#333333")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "xai_pipeline_diagram.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[diagram] wrote {out}")


def fig_resnet_backbone():
    fig, ax = plt.subplots(figsize=(9.5, 12.5))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 15.5)
    ax.axis("off")

    ax.text(0.2, 15.15, "(b) ", fontsize=19, fontweight="bold", ha="left", va="top")
    ax.text(0.85, 15.15, "ResNet-50  Feature Extraction  +  Grad-CAM",
             fontsize=17.5, fontweight="bold", ha="left", va="top")

    cx = 4.6
    w = 8.7
    stages = [
        ("Input Image", "224×224×3  ultrasound frame", "#263238"),
        ("Conv1  +  BN  +  ReLU", "7×7, stride 2, 64 channels", "#37474f"),
        ("Max Pooling", "3×3, stride 2", "#546e7a"),
        ("Stage 1  (3× ResBlock)", "Bottleneck  64×4 = 256 ch", "#0d47a1"),
        ("Stage 2  (4× ResBlock)", "Bottleneck 128×4 = 512 ch", "#1565c0"),
        ("Stage 3  (6× ResBlock)", "Bottleneck 256×4 = 1024 ch", "#1976d2"),
    ]

    y = 14.15
    step = 1.55
    for title, sub, color in stages:
        _box(ax, cx, y, w, 1.15, color, title, sub, title_size=15, sub_size=11.5)
        _arrow(ax, cx, y - 0.6, cx, y - step + 0.6)
        y -= step

    # Stage 4 with CAM tap highlighted
    stage4_y = y
    box4 = FancyBboxPatch((cx - w / 2, stage4_y - 0.6), w, 1.2,
                           boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=2.6, edgecolor="#e65100", facecolor="#1e88e5", zorder=2)
    ax.add_patch(box4)
    ax.text(cx, stage4_y + 0.19, "Stage 4  (3× ResBlock)", ha="center", va="center",
             fontsize=15, fontweight="bold", color="white", zorder=3)
    ax.text(cx, stage4_y - 0.25, "Bottleneck 512×4 = 2048 ch", ha="center", va="center",
             fontsize=11.5, style="italic", color="white", zorder=3)

    cam_x = cx + w / 2 + 0.85
    cam_box = FancyBboxPatch((cam_x - 0.6, stage4_y - 0.35), 1.2, 0.7,
                              boxstyle="round,pad=0.02,rounding_size=0.02",
                              linewidth=1.6, edgecolor="#1a1a1a", facecolor="#e65100", zorder=2)
    ax.add_patch(cam_box)
    ax.text(cam_x, stage4_y, "CAM", ha="center", va="center", fontsize=13,
             fontweight="bold", color="white", zorder=3)
    _arrow(ax, cx + w / 2 + 0.05, stage4_y, cam_x - 0.6, stage4_y, color="#e65100")
    ax.text(cam_x, stage4_y - 0.85, "Grad-CAM\nheatmap", ha="center", va="top",
             fontsize=10.5, style="italic", color="#e65100")

    _arrow(ax, cx, stage4_y - 0.6, cx, stage4_y - step + 0.6)
    y = stage4_y - step

    _box(ax, cx, y, w, 1.15, "#1e88e5", "Global Avg. Pooling", "2048-d feature vector",
         title_size=15, sub_size=11.5)
    _arrow(ax, cx, y - 0.6, cx, y - step + 0.6)
    y -= step

    _box(ax, cx, y, w, 1.15, "#00695c", "FC  +  Softmax", "9 disease classes",
         title_size=15, sub_size=11.5)

    # output ellipses
    n = len(SHORT_CODES)
    colors = ["#2e7d32", "#1565c0", "#c1560c", "#6a1b9a", "#b71c1c",
              "#00838f", "#8d6e63", "#ad1457", "#4527a0"]
    ell_y = y - 1.35
    span = w
    xs = [cx - span / 2 + span * (i + 0.5) / n for i in range(n)]
    for i, (code, col, ex) in enumerate(zip(SHORT_CODES, colors, xs)):
        _arrow(ax, cx, y - 0.6, ex, ell_y + 0.32, color="#616161", lw=1.1)
        e = Ellipse((ex, ell_y), width=span / n * 0.86, height=0.62,
                     facecolor=col, edgecolor="#1a1a1a", linewidth=1.2, zorder=2)
        ax.add_patch(e)
        ax.text(ex, ell_y, code, ha="center", va="center", fontsize=10.5,
                 fontweight="bold", color="white", zorder=3,
                 path_effects=[withStroke(linewidth=0)])

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "resnet_backbone.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[diagram] wrote {out}")


if __name__ == "__main__":
    fig_xai_pipeline()
    fig_resnet_backbone()
