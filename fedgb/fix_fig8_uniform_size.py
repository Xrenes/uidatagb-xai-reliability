"""
fix_fig8_uniform_size.py
---------------------------
Regenerates Fig. 8's two failure-case images (failure_case_perforation.png,
failure_case_gallstones.png) at a fixed, identical figure size so both
render at the same aspect ratio when displayed side by side at
width:340px in the paper -- fixes a mismatch caused by
analyze_failure_cases_uidatagb.py's use of save_gradcam_figure(), whose
bbox_inches="tight" + variable-length "[failure case] Pred:/True:" title
produces a different output width per class.

Reuses the exact validation-set indices already recorded in
outputs/phase0/failure_cases.json (index 1780 = Perforation-as-Gallstones,
index 272 = Gallstones-as-Abdomen&Retroperitoneum), so the same two
images already described in the paper's text/caption are reproduced,
not new ones.

Run from fedgb/:  python fix_fig8_uniform_size.py
Outputs: outputs/figures/failure_case_perforation.png (overwritten)
         outputs/figures/failure_case_gallstones.png (overwritten)
"""
import json
import os

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CLASS_NAMES, CLASS_LABELS, GBCUDataset, get_transforms
from evaluate import DEVICE
from gradcam import GradCAMPlusPlus, overlay_heatmap
from model import build_model, get_target_layer

REPO = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(REPO, "outputs", "seeds", "seed_42", "centralized.pt")
DATA_DIR = os.path.join(REPO, "data", "uidatagb", "validation")
FIG_DIR = os.path.join(REPO, "outputs", "figures")
JSON_PATH = os.path.join(REPO, "outputs", "phase0", "failure_cases.json")

INV = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                           std=[1/0.229, 1/0.224, 1/0.225])

FIG_W_IN, FIG_H_IN, DPI = 12.0, 4.13, 150  # identical to generate_supp_a_3panel.py


def save_fixed_size_3panel(original_image, heatmap, pred_label, true_label,
                            confidence, save_path):
    overlay = overlay_heatmap(original_image, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_IN, FIG_H_IN))

    axes[0].imshow(original_image.resize((224, 224)))
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM++ (raw)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay — Pred: {pred_label} ({confidence:.1%})\nTrue: {true_label}",
                      fontsize=11)
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    val = GBCUDataset(DATA_DIR, transform=get_transforms(train=False))
    cam = GradCAMPlusPlus(model, get_target_layer(model))
    os.makedirs(FIG_DIR, exist_ok=True)

    for case in data["highlighted_cases"]:
        idx = case["index"]
        tensor, label = val[idx]
        heat, pred_cam, conf_cam = cam.generate(tensor.unsqueeze(0).to(DEVICE))
        orig = np.clip(INV(tensor).permute(1, 2, 0).numpy(), 0, 1)
        original_image = Image.fromarray((orig * 255).astype(np.uint8))

        slug = case["true_class"].split("_", 1)[1]
        save_path = os.path.join(FIG_DIR, f"failure_case_{slug}.png")
        save_fixed_size_3panel(
            original_image=original_image,
            heatmap=heat,
            pred_label=case["pred_class_label"],
            true_label=case["true_class_label"],
            confidence=case["confidence"],
            save_path=save_path,
        )
        print(f"[fix-fig8] wrote {save_path}")


if __name__ == "__main__":
    main()
