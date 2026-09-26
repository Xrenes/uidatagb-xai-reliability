"""
stage2_ablation_fixes.py
--------------------------
Two cheap, targeted attempts to recover genuine cross-batch generalisation
on the Stage-2 (batch-corrected) split, without a large new effort
(no domain-adversarial training). Baseline to beat: seed=42 primary model,
accuracy=0.3100, macro-F1=0.2944, macro-AUC=0.7351 (Section 4.5).

  Experiment A: per-image normalisation instead of fixed ImageNet mean/std.
  Rationale: if the batch shortcut rides on global brightness/contrast/gain
  statistics (plausible -- the batch-ID classifier's Grad-CAM attended to
  central content, not borders, so it isn't a crude spatial artefact),
  normalising each image by its own mean/std removes exactly that kind of
  global per-image statistic before the network ever sees it.

  Experiment B: stronger, statistics-disrupting augmentation (heavier
  color jitter, random autocontrast, random equalize, random sharpness)
  on top of standard ImageNet normalisation. Rationale: forces the model
  away from features that don't survive aggressive appearance perturbation,
  which a low-level batch/scanner signature likely would not.

Both use the identical architecture, optimizer, and epoch budget as the
Stage-2 primary model (Table 2, Section 3.2) -- only the transform
pipeline changes -- so any accuracy difference is attributable to the
transform, not a confounded hyperparameter change.

Run from fedgb/:  python stage2_ablation_fixes.py
Outputs: outputs/session_corrected/ablation/{per_image_norm,stronger_aug}.pt
         outputs/session_corrected/ablation/{per_image_norm,stronger_aug}_metrics.json
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from dataset import CLASS_NAMES, CLASS_TO_IDX
from model import build_model
from evaluate import run_inference, compute_metrics

DATA_DIR = "./data/uidatagb_sessioncorrected"
OUT_DIR = "./outputs/session_corrected/ablation"
SEED = 42
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PerImageNormalize:
    """Normalizes each image by its own per-channel mean/std instead of a
    fixed dataset-wide constant, removing global per-image brightness/
    contrast/gain statistics before the network sees them."""
    def __call__(self, tensor):
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return (tensor - mean) / std


def get_transform_per_image_norm(train):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)]
    ops += [transforms.ToTensor(), PerImageNormalize()]
    return transforms.Compose(ops)


def get_transform_stronger_aug(train):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomEqualize(p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        ]
    ops += [transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(ops)


class ImageFolderDS(Dataset):
    def __init__(self, data_dir, transform):
        self.samples = []
        for cname in CLASS_NAMES:
            cdir = os.path.join(data_dir, cname)
            if not os.path.isdir(cdir):
                continue
            label = CLASS_TO_IDX[cname]
            for fname in os.listdir(cdir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(cdir, fname), label))
        self.transform = transform
        print(f"[ablation] loaded {len(self.samples)} images from {data_dir}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def run_experiment(tag, train_tf_fn, val_tf_fn):
    t0 = time.time()
    print(f"\n[ablation] === {tag} ===", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_ds = ImageFolderDS(os.path.join(DATA_DIR, "training"), train_tf_fn(train=True))
    val_ds = ImageFolderDS(os.path.join(DATA_DIR, "validation"), val_tf_fn(train=False))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_model(pretrained=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
        print(f"[{tag}] epoch {epoch+1}/{EPOCHS} loss={running_loss/len(train_ds):.4f} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    model.eval()
    y_true, y_pred, y_probs = run_inference(model, val_loader)
    metrics = compute_metrics(y_true, y_pred, y_probs)
    print(f"[{tag}] accuracy={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} "
          f"auc={metrics['auc_macro']:.4f}  (baseline: 0.3100 / 0.2944 / 0.7351)", flush=True)

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{tag}.pt"))
    result = {"tag": tag, "accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"],
              "auc_macro": metrics["auc_macro"],
              "baseline_accuracy": 0.30998659517426275,
              "baseline_f1_macro": 0.29438966058250315,
              "baseline_auc_macro": 0.7351375517029008,
              "delta_accuracy": metrics["accuracy"] - 0.30998659517426275}
    with open(os.path.join(OUT_DIR, f"{tag}_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] wrote {OUT_DIR}/{tag}.pt ({time.time()-t0:.0f}s total)", flush=True)
    return result


def main():
    r1 = run_experiment("per_image_norm", get_transform_per_image_norm, get_transform_per_image_norm)
    r2 = run_experiment("stronger_aug", get_transform_stronger_aug, get_transform_stronger_aug)

    print("\n[ablation] === SUMMARY ===", flush=True)
    print(f"  baseline (Table 10)      : accuracy=0.3100 f1=0.2944 auc=0.7351", flush=True)
    print(f"  per_image_norm           : accuracy={r1['accuracy']:.4f} f1={r1['f1_macro']:.4f} "
          f"auc={r1['auc_macro']:.4f}  (delta={r1['delta_accuracy']:+.4f})", flush=True)
    print(f"  stronger_aug              : accuracy={r2['accuracy']:.4f} f1={r2['f1_macro']:.4f} "
          f"auc={r2['auc_macro']:.4f}  (delta={r2['delta_accuracy']:+.4f})", flush=True)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"per_image_norm": r1, "stronger_aug": r2}, f, indent=2)


if __name__ == "__main__":
    main()
