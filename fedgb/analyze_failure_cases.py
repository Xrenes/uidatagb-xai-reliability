"""
analyze_failure_cases.py
--------------------------
Real, grounded failure-case analysis for the paper's Results section: runs
fresh inference with the seed-42 global FedProx model (round_010.pt),
finds concrete misclassified validation images for the two weakest classes
(BMT, Malignant), and renders their Grad-CAM++ heatmaps so the paper can
discuss *specific* wrong predictions instead of only aggregate metrics.

Run from fedgb/:
  python analyze_failure_cases.py

Outputs:
  outputs/figures/failure_case_<n>.png   (3-panel Grad-CAM++ figure)
  outputs/failure_cases.json             (structured facts for the writeup)
"""
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CLASS_NAMES, GBCUDataset, get_transforms
import evaluate
from evaluate import compute_metrics, run_inference
from gradcam import GradCAMPlusPlus, save_gradcam_figure
from model import build_model, get_target_layer

# evaluate.py auto-selects CUDA at import time regardless of the model's
# actual device; force it to CPU here since the GPU is occupied by the
# background training sweep and we don't want to contend for its VRAM.
evaluate.DEVICE = torch.device("cpu")

REPO = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(REPO, "outputs", "checkpoints", "round_010.pt")
DATA_DIR = os.path.join(REPO, "data", "data", "validation")
FIG_DIR = os.path.join(REPO, "outputs", "figures")
OUT_JSON = os.path.join(REPO, "outputs", "failure_cases.json")

INV = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                           std=[1/0.229, 1/0.224, 1/0.225])


def main():
    # Forced CPU: GPU is occupied by the background multi-seed sweep, and
    # this workload (689-image single pass + 2 heatmaps) is cheap on CPU.
    device = torch.device("cpu")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    val = GBCUDataset(DATA_DIR, transform=get_transforms(train=False))
    loader = DataLoader(val, batch_size=16, shuffle=False, num_workers=0)
    y_true, y_pred, y_probs = run_inference(model, loader)
    metrics = compute_metrics(y_true, y_pred, y_probs)
    cm = metrics["confusion_matrix"]
    print("[info] confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Find misclassified indices per class, ranked by model confidence
    # (highest-confidence wrong answers are the most interesting failures)
    wrong = [(i, int(y_true[i]), int(y_pred[i]), float(y_probs[i][y_pred[i]]))
             for i in range(len(y_true)) if y_true[i] != y_pred[i]]

    cam = GradCAMPlusPlus(model, get_target_layer(model))
    os.makedirs(FIG_DIR, exist_ok=True)
    cases = []

    for target_true_cls in ("bmt", "malg"):
        cls_idx = CLASS_NAMES.index(target_true_cls)
        candidates = sorted(
            [w for w in wrong if w[1] == cls_idx],
            key=lambda w: -w[3],  # most confident wrong answer first
        )
        if not candidates:
            continue
        idx, true_c, pred_c, conf = candidates[0]
        tensor, label = val[idx]
        heat, pred_cam, conf_cam = cam.generate(tensor.unsqueeze(0))
        orig = np.clip(INV(tensor).permute(1, 2, 0).numpy(), 0, 1)
        original_image = Image.fromarray((orig * 255).astype(np.uint8))
        save_path = os.path.join(FIG_DIR, f"failure_case_{target_true_cls}.png")
        save_gradcam_figure(
            original_image=original_image, heatmap=heat,
            pred_class=pred_cam, confidence=conf_cam, true_class=label,
            save_path=save_path, title_prefix="[failure case] ",
            heatmap_label="Grad-CAM++ Heatmap",
        )
        src_file = os.path.basename(val.samples[idx][0])
        cases.append({
            "true_class": CLASS_NAMES[true_c],
            "pred_class": CLASS_NAMES[pred_c],
            "confidence": round(conf, 3),
            "src_file": src_file,
            "figure": os.path.relpath(save_path, REPO).replace("\\", "/"),
            "index": idx,
        })
        print(f"[case] true={CLASS_NAMES[true_c]} pred={CLASS_NAMES[pred_c]} "
              f"conf={conf:.3f} file={src_file} -> {save_path}")

    cam.remove_hooks()

    # Also report the dominant confusion pairs overall (for prose accuracy)
    pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i][j] > 0:
                pairs.append({"true": CLASS_NAMES[i], "pred": CLASS_NAMES[j],
                              "count": int(cm[i][j])})
    pairs.sort(key=lambda p: -p["count"])

    result = {
        "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
        "class_names": CLASS_NAMES,
        "top_confusion_pairs": pairs[:6],
        "highlighted_cases": cases,
        "n_val": len(val),
        "n_wrong_total": len(wrong),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] wrote {OUT_JSON}")
    print("[top confusion pairs]")
    for p in pairs[:6]:
        print(f"  true={p['true']:5s} pred={p['pred']:5s} n={p['count']}")


if __name__ == "__main__":
    main()
