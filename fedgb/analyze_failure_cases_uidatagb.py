"""
analyze_failure_cases_uidatagb.py
------------------------------------
Grounded failure-case analysis for the UIdataGB primary model: finds
concrete misclassified validation images for the two weakest classes
(Perforation, Gallstones -- F1 0.894/0.900, the lowest of the 9), and
renders their Grad-CAM++ heatmaps.

Run from fedgb/:  python analyze_failure_cases_uidatagb.py
Outputs:
  outputs/figures/failure_case_<class>.png   (3-panel Grad-CAM++ figure)
  outputs/phase0/failure_cases.json
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset import CLASS_NAMES, CLASS_LABELS, GBCUDataset, get_transforms
from evaluate import compute_metrics, run_inference, DEVICE
from gradcam import GradCAMPlusPlus, save_gradcam_figure
from model import build_model, get_target_layer

REPO = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get(
    "P0_CKPT", os.path.join(REPO, "outputs", "seeds", "seed_42", "centralized.pt"))
DATA_DIR = os.environ.get(
    "P0_VAL_DIR", os.path.join(REPO, "data", "uidatagb", "validation"))
FIG_DIR = os.environ.get(
    "P0_FIG_DIR", os.path.join(REPO, "outputs", "figures"))
OUT_JSON = os.environ.get(
    "P0_FAILURE_JSON", os.path.join(REPO, "outputs", "phase0", "failure_cases.json"))

# TARGET_CLASSES is resolved at runtime to the two weakest-F1 classes for
# whichever checkpoint/split is loaded (Stage-1's weakest classes are not
# assumed to still be weakest on the Stage-2, batch-corrected split).
TARGET_CLASSES = None

INV = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                           std=[1/0.229, 1/0.224, 1/0.225])


def main():
    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    val = GBCUDataset(DATA_DIR, transform=get_transforms(train=False))
    loader = DataLoader(val, batch_size=32, shuffle=False, num_workers=0)
    y_true, y_pred, y_probs = run_inference(model, loader)
    metrics = compute_metrics(y_true, y_pred, y_probs)
    cm = metrics["confusion_matrix"]
    print("[info] confusion matrix computed")

    global TARGET_CLASSES
    if TARGET_CLASSES is None:
        f1_per_class = metrics.get("f1_per_class")
        if f1_per_class is None:
            f1_per_class = metrics.get("per_class_f1")
        if f1_per_class is not None:
            order = sorted(range(len(CLASS_NAMES)), key=lambda i: f1_per_class[i])
            TARGET_CLASSES = [CLASS_NAMES[order[0]], CLASS_NAMES[order[1]]]
            print(f"[info] auto-detected two weakest classes by F1: "
                  f"{TARGET_CLASSES[0]} (F1={f1_per_class[order[0]]:.4f}), "
                  f"{TARGET_CLASSES[1]} (F1={f1_per_class[order[1]]:.4f})")
        else:
            TARGET_CLASSES = ["05_perforation", "01_gallstones"]
            print("[warn] per-class F1 not found in metrics; falling back to "
                  "Stage-1's weakest classes")

    wrong = [(i, int(y_true[i]), int(y_pred[i]), float(y_probs[i][y_pred[i]]))
             for i in range(len(y_true)) if y_true[i] != y_pred[i]]

    cam = GradCAMPlusPlus(model, get_target_layer(model))
    os.makedirs(FIG_DIR, exist_ok=True)
    cases = []

    for target_true_cls in TARGET_CLASSES:
        cls_idx = CLASS_NAMES.index(target_true_cls)
        candidates = sorted(
            [w for w in wrong if w[1] == cls_idx],
            key=lambda w: -w[3],
        )
        if not candidates:
            print(f"[warn] no misclassifications found for {target_true_cls}")
            continue
        idx, true_c, pred_c, conf = candidates[0]
        tensor, label = val[idx]
        heat, pred_cam, conf_cam = cam.generate(tensor.unsqueeze(0).to(DEVICE))
        orig = np.clip(INV(tensor).permute(1, 2, 0).numpy(), 0, 1)
        original_image = Image.fromarray((orig * 255).astype(np.uint8))
        slug = target_true_cls.split("_", 1)[1]
        save_path = os.path.join(FIG_DIR, f"failure_case_{slug}.png")
        save_gradcam_figure(
            original_image=original_image, heatmap=heat,
            pred_class=pred_cam, confidence=conf_cam, true_class=label,
            save_path=save_path, title_prefix="[failure case] ",
            heatmap_label="Grad-CAM++ Heatmap",
        )
        src_file = os.path.basename(val.samples[idx][0])
        cases.append({
            "true_class": CLASS_NAMES[true_c],
            "true_class_label": CLASS_LABELS[CLASS_NAMES[true_c]],
            "pred_class": CLASS_NAMES[pred_c],
            "pred_class_label": CLASS_LABELS[CLASS_NAMES[pred_c]],
            "confidence": round(conf, 3),
            "src_file": src_file,
            "figure": os.path.relpath(save_path, REPO).replace("\\", "/"),
            "index": idx,
        })
        print(f"[case] true={CLASS_LABELS[CLASS_NAMES[true_c]]} "
              f"pred={CLASS_LABELS[CLASS_NAMES[pred_c]]} conf={conf:.3f} "
              f"file={src_file} -> {save_path}")

    cam.remove_hooks()

    pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i][j] > 0:
                pairs.append({"true": CLASS_LABELS[CLASS_NAMES[i]],
                              "pred": CLASS_LABELS[CLASS_NAMES[j]],
                              "count": int(cm[i][j])})
    pairs.sort(key=lambda p: -p["count"])

    result = {
        "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
        "class_names": CLASS_NAMES,
        "top_confusion_pairs": pairs[:8],
        "highlighted_cases": cases,
        "n_val": len(val),
        "n_wrong_total": len(wrong),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] wrote {OUT_JSON}")
    print("[top confusion pairs]")
    for p in pairs[:8]:
        print(f"  true={p['true']:38s} pred={p['pred']:38s} n={p['count']}")


if __name__ == "__main__":
    main()
