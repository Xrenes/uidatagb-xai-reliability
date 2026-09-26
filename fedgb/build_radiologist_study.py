"""
build_radiologist_study.py
--------------------------
Prepares a radiologist validation study for the XAI-FedGB Grad-CAM++
heatmaps. Selects a stratified sample of validation images, renders the
global-model Grad-CAM++ overlay for each, and emits a blank rating template
(CSV) plus a manifest. Two (or more) radiologists independently fill the
template; compute_kappa.py then reports inter-rater agreement and correlation
with the cosine-overlap flags.

Stratification: N_PER_CLASS images per disease class, chosen to span the
model's confidence range and to include both correctly and incorrectly
classified cases (so plausibility ratings are not confounded with accuracy).

Run from fedgb/:
  python build_radiologist_study.py

Outputs (under outputs/radiologist_study/):
  panels/<class>_<n>_<correct|wrong>.png   Grad-CAM++ overlay panels to rate
  rating_template.csv                        blank Likert form (one row/image)
  manifest.json                              ground-truth + prediction per image
"""
from __future__ import annotations
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CLASS_NAMES, GBCUDataset, get_transforms
from gradcam import GradCAMPlusPlus, overlay_heatmap
from model import build_model, get_target_layer
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PER_CLASS = 10          # 10 x 5 = 50 images total
GLOBAL_CKPT = "./outputs/checkpoints/round_010.pt"
OUT = "./outputs/radiologist_study"

INV = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                           std=[1/0.229, 1/0.224, 1/0.225])


def main():
    os.makedirs(os.path.join(OUT, "panels"), exist_ok=True)
    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(GLOBAL_CKPT, map_location=DEVICE))
    model.eval()
    cam = GradCAMPlusPlus(model, get_target_layer(model))

    val = GBCUDataset("./data/data/validation", transform=get_transforms(train=False))

    # group indices by class, record prediction + confidence
    by_class = {c: [] for c in range(len(CLASS_NAMES))}
    for idx in range(len(val)):
        tensor, label = val[idx]
        with torch.no_grad():
            p = torch.softmax(model(tensor.unsqueeze(0).to(DEVICE)), 1)[0]
        conf, pred = float(p.max()), int(p.argmax())
        by_class[label].append((idx, pred, conf))

    rng = np.random.default_rng(42)
    manifest = []
    rows = []
    rated_id = 0
    for c in range(len(CLASS_NAMES)):
        items = by_class[c]
        wrong = [t for t in items if t[1] != c]
        right = [t for t in items if t[1] == c]
        # aim for a mix: up to 40% wrong (or as many as exist), rest correct
        n_wrong = min(len(wrong), max(1, int(0.4 * N_PER_CLASS)))
        n_right = N_PER_CLASS - n_wrong
        pick = []
        if wrong:
            pick += [wrong[i] for i in rng.choice(len(wrong), min(n_wrong, len(wrong)), replace=False)]
        if right:
            pick += [right[i] for i in rng.choice(len(right), min(n_right, len(right)), replace=False)]
        for n, (idx, pred, conf) in enumerate(pick, 1):
            tensor, label = val[idx]
            heat, _, _ = cam.generate(tensor.unsqueeze(0))
            orig = np.clip(INV(tensor).permute(1, 2, 0).numpy(), 0, 1)
            ov = overlay_heatmap(Image.fromarray((orig*255).astype(np.uint8)), heat)
            correct = (pred == label)
            fn = f"{CLASS_NAMES[c]}_{n:02d}_{'correct' if correct else 'wrong'}.png"
            Image.fromarray(ov).save(os.path.join(OUT, "panels", fn))
            rated_id += 1
            manifest.append({"rating_id": rated_id, "panel": fn,
                             "true_class": CLASS_NAMES[label],
                             "pred_class": CLASS_NAMES[pred],
                             "model_correct": bool(correct),
                             "model_confidence": round(conf, 3),
                             "src": os.path.basename(val.samples[idx][0])})
            rows.append(rated_id)

    cam.remove_hooks()

    # blank rating template — radiologist fills 'plausibility_1to5' and 'attends_gb_region_yn'
    with open(os.path.join(OUT, "rating_template.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rating_id", "panel", "plausibility_1to5",
                    "attends_gb_region_yn", "notes"])
        for m in manifest:
            w.writerow([m["rating_id"], m["panel"], "", "", ""])

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump({"n_images": len(manifest), "n_per_class": N_PER_CLASS,
                   "classes": CLASS_NAMES, "items": manifest}, f, indent=2)

    n_wrong = sum(1 for m in manifest if not m["model_correct"])
    print(f"[done] {len(manifest)} panels ({n_wrong} misclassified, "
          f"{len(manifest)-n_wrong} correct) -> {OUT}/panels/")
    print(f"[done] rating_template.csv and manifest.json written to {OUT}/")


if __name__ == "__main__":
    main()
