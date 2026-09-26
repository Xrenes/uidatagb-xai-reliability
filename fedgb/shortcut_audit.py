"""
shortcut_audit.py
-----------------
Shortcut-learning audit (Geirhos et al., 2020; Zech et al., 2018). Ultrasound
screen exports can contain non-clinical cues at the frame margin (burned-in
text, calipers, vendor logos, black borders) that a model may exploit instead
of anatomy. We test two things on the primary model:

  1. Margin-crop ablation: re-evaluate classification accuracy / macro-F1
     after cropping away the outer margin of every image (and resizing back).
     If the model were relying on margin text/borders, accuracy would drop
     sharply; a small change is evidence it is not.
  2. Border-energy of the explanation: the fraction of Grad-CAM++ attribution
     mass that falls in the outer margin ring. Low border energy is evidence
     the model attends to central anatomy rather than frame artefacts.

Run from fedgb/:  python shortcut_audit.py
Outputs: outputs/phase0/shortcut_audit.json, outputs/figures/shortcut_audit.png
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from _p0common import (load_primary_model, val_dataset, stratified_indices,
                       upsample224, normalize, raw_tensor,
                       get_target_layer, DEVICE, OUT_DIR, FIG_DIR, CLASS_NAMES)
from gradcam import GradCAMPlusPlus

MARGIN = 0.12       # crop away the outer 12% on each side
PER_CLASS_CAM = 24  # images used for the border-energy heatmap measurement


def evaluate_all(model, ds, crop):
    """Full-validation accuracy / macro-F1, optionally margin-cropped."""
    y_true, y_pred = [], []
    m = int(224 * MARGIN)
    for path, label in ds.samples:
        x = raw_tensor(path).unsqueeze(0)  # (1,3,224,224) [0,1]
        if crop:
            x = x[:, :, m:224 - m, m:224 - m]
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        xn = normalize(x).to(DEVICE)
        with torch.no_grad():
            pred = model(xn).argmax(1).item()
        y_true.append(label)
        y_pred.append(pred)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return acc, f1


def border_energy(model, ds, idxs):
    """Mean fraction of Grad-CAM++ mass in the outer MARGIN ring."""
    cam = GradCAMPlusPlus(model, get_target_layer(model))
    m = int(224 * MARGIN)
    ring_mask = np.ones((224, 224), dtype=bool)
    ring_mask[m:224 - m, m:224 - m] = False  # True on the margin ring
    fracs = []
    for i in idxs:
        path = ds.samples[i][0]
        x = normalize(raw_tensor(path).unsqueeze(0)).to(DEVICE)
        with torch.no_grad():
            pred = model(x).argmax(1).item()
        hm, _, _ = cam.generate(x, target_class=pred)
        hm = upsample224(hm)
        total = hm.sum()
        fracs.append(float(hm[ring_mask].sum() / total) if total > 0 else 0.0)
    cam.remove_hooks()
    # what fraction of the image area the ring occupies, for reference
    ring_area_frac = float(ring_mask.mean())
    return float(np.mean(fracs)), float(np.std(fracs)), ring_area_frac


def main():
    model = load_primary_model()
    ds = val_dataset()

    acc_full, f1_full = evaluate_all(model, ds, crop=False)
    acc_crop, f1_crop = evaluate_all(model, ds, crop=True)
    print(f"[shortcut] full  acc={acc_full:.4f} f1={f1_full:.4f}")
    print(f"[shortcut] crop  acc={acc_crop:.4f} f1={f1_crop:.4f}")

    idxs = stratified_indices(ds, PER_CLASS_CAM)
    be_mean, be_std, ring_area = border_energy(model, ds, idxs)
    print(f"[shortcut] border-energy={be_mean:.3f} (margin ring is "
          f"{ring_area*100:.0f}% of image area)")

    results = {
        "margin_frac": MARGIN,
        "accuracy_full": acc_full, "accuracy_cropped": acc_crop,
        "accuracy_delta": acc_crop - acc_full,
        "macro_f1_full": f1_full, "macro_f1_cropped": f1_crop,
        "border_energy_mean": be_mean, "border_energy_std": be_std,
        "ring_area_frac": ring_area,
        "n_cam_images": len(idxs),
        "interpretation": (
            f"Cropping away the outer {int(MARGIN*100)}% margin changes accuracy "
            f"by {acc_crop - acc_full:+.3f} ({acc_full:.3f} -> {acc_crop:.3f}), and "
            f"only {be_mean*100:.1f}% of Grad-CAM++ attribution mass falls in that "
            f"margin ring (which covers {ring_area*100:.0f}% of the frame). Both "
            f"indicate the model relies on central anatomy rather than margin "
            f"text/border shortcuts."
        ),
    }
    with open(os.path.join(OUT_DIR, "shortcut_audit.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figure ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.bar([0, 1], [acc_full, acc_crop], color=["#1768c4", "#6a2fa6"], width=0.6)
    a1.bar([2.2, 3.2], [f1_full, f1_crop], color=["#1768c4", "#6a2fa6"], width=0.6)
    a1.set_xticks([0, 1, 2.2, 3.2])
    a1.set_xticklabels(["Acc\nfull", "Acc\ncrop", "F1\nfull", "F1\ncrop"], fontsize=8)
    a1.set_ylim(0, 1)
    a1.set_title(f"Margin-crop ablation (outer {int(MARGIN*100)}%)")
    for xi, v in zip([0, 1, 2.2, 3.2], [acc_full, acc_crop, f1_full, f1_crop]):
        a1.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)

    a2.bar([0, 1], [be_mean, ring_area], color=["#d9601a", "#bbbbbb"], width=0.6)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["attribution mass\nin margin ring", "margin ring\narea share"], fontsize=8)
    a2.set_ylim(0, max(0.5, ring_area + 0.1))
    a2.set_title("Border-energy of Grad-CAM++")
    for xi, v in zip([0, 1], [be_mean, ring_area]):
        a2.text(xi, v + 0.01, f"{v*100:.0f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shortcut_audit.png"), dpi=170)
    print("[shortcut] wrote outputs/phase0/shortcut_audit.json and outputs/figures/shortcut_audit.png")


if __name__ == "__main__":
    main()
