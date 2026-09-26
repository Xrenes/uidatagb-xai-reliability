"""
batch_id_shortcut_diagnostic.py
---------------------------------
Direct test of the Stage-2 batch-correlated-shortcut hypothesis
(Section 3.1 / Limitation 1): if the drop from 93.05% (Stage 1) to 29.59%
(Stage 2) is caused by a learnable, non-anatomical signature that is
consistent within one export/acquisition batch, then a classifier should
be able to tell batches apart EVEN WHEN THE DISEASE LABEL IS HELD FIXED
-- since disease label cannot explain batch identity within a single
class, any batch-discrimination accuracy well above chance is direct
evidence of a non-anatomical, batch-correlated signature.

Design: restrict to the single largest disease class (08_carcinoma), take
its K largest acquisition batches (by filename-prefix, see Section 3.1),
and train a K-way "which batch is this image from" classifier on an
ordinary (non-leakage-audited) image-level split of those same batches'
images. High held-out accuracy on this task, despite every image sharing
one disease label, is direct proof a learnable batch signature exists.
Grad-CAM on the resulting classifier visualises what it keys on.

Run from fedgb/:  python batch_id_shortcut_diagnostic.py
Outputs: outputs/phase0/batch_id_diagnostic.json
         outputs/figures/batch_id_diagnostic.png       (accuracy + confusion)
         outputs/figures/batch_id_gradcam_gallery.png  (what the shortcut looks like)
"""
import json
import os
import random
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import build_model, get_target_layer
from gradcam import GradCAMPlusPlus, overlay_heatmap

DATA_DIR = "./data/uidatagb"
TARGET_CLASS = "08_carcinoma"          # largest class, most batches to choose from
N_BATCHES = 10                          # K-way batch-ID task
MIN_BATCH_SIZE = 25                     # only consider batches with enough images
VAL_FRAC = 0.30
SEED = 42
EPOCHS = 6
OUT_DIR = "./outputs/phase0"
FIG_DIR = "./outputs/figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FNAME_RE = re.compile(r'^([A-Za-z]+\d+)\s*\(\d+\)\.\w+$')

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class BatchIDDataset(Dataset):
    def __init__(self, items, train):
        self.items = items  # list of (path, batch_label_idx)
        tf = [transforms.Resize((224, 224))]
        if train:
            tf += [transforms.RandomHorizontalFlip(),
                   transforms.RandomRotation(15),
                   transforms.ColorJitter(brightness=0.2, contrast=0.2)]
        tf += [transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(tf)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def collect_batches():
    items_by_prefix = defaultdict(list)
    for split in ("training", "validation"):
        cdir = os.path.join(DATA_DIR, split, TARGET_CLASS)
        if not os.path.isdir(cdir):
            continue
        for fname in os.listdir(cdir):
            m = FNAME_RE.match(fname)
            if m:
                items_by_prefix[m.group(1).lower()].append(os.path.join(cdir, fname))
    return items_by_prefix


def main():
    t0 = time.time()
    items_by_prefix = collect_batches()
    eligible = {k: v for k, v in items_by_prefix.items() if len(v) >= MIN_BATCH_SIZE}
    print(f"[batch-id] {len(items_by_prefix)} total batches in {TARGET_CLASS}, "
          f"{len(eligible)} with >= {MIN_BATCH_SIZE} images", flush=True)

    top_batches = sorted(eligible.items(), key=lambda kv: -len(kv[1]))[:N_BATCHES]
    batch_names = [k for k, _ in top_batches]
    print(f"[batch-id] using {len(batch_names)} batches: "
          f"{[(k, len(v)) for k, v in top_batches]}", flush=True)

    rng = random.Random(SEED)
    train_items, val_items = [], []
    for label_idx, (bname, paths) in enumerate(top_batches):
        paths = paths[:]
        rng.shuffle(paths)
        n_val = max(1, round(len(paths) * VAL_FRAC))
        val_items += [(p, label_idx) for p in paths[:n_val]]
        train_items += [(p, label_idx) for p in paths[n_val:]]

    print(f"[batch-id] train={len(train_items)} val={len(val_items)} images "
          f"(image-level split within each batch -- this task deliberately "
          f"is NOT leakage-audited, since it tests same-batch discriminability, "
          f"not generalisation to unseen batches)", flush=True)

    train_ds = BatchIDDataset(train_items, train=True)
    val_ds = BatchIDDataset(val_items, train=False)
    # num_workers=0: on this machine, num_workers>0 (separate spawned
    # processes on Windows) caused disk-I/O contention and erratic,
    # sometimes much SLOWER epochs rather than a speedup -- reverted
    # after observing this in session_extra_seeds.py's live run.
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    torch.manual_seed(SEED)
    model = build_model(pretrained=True).to(DEVICE)
    # swap the 9-class head for a K-way batch-ID head
    in_features = model.fc[1].in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(in_features, 256), nn.ReLU(),
        nn.Dropout(p=0.3), nn.Linear(256, len(batch_names)),
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
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
        print(f"[batch-id] epoch {epoch+1}/{EPOCHS} loss={running_loss/len(train_ds):.4f} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    model.eval()
    correct, total = 0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            pred = model(x).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_true += y.tolist()
            all_pred += pred.tolist()
    acc = correct / total
    chance = 1.0 / len(batch_names)
    print(f"\n[batch-id] === RESULT ===\n"
          f"  {len(batch_names)}-way batch-ID accuracy = {acc:.4f}  "
          f"(chance = {chance:.4f}, {acc/chance:.1f}x chance)\n"
          f"  All images share ONE disease label ({TARGET_CLASS}); this "
          f"accuracy is therefore attributable to a non-anatomical, "
          f"batch-correlated signature, not diagnosis.", flush=True)

    # confusion matrix figure
    K = len(batch_names)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(["Batch-ID\nclassifier", "Chance\nbaseline"], [acc, chance],
              color=["#c0392b", "#888888"])
    ax[0].set_ylim(0, 1.15)
    ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[0].set_ylabel("Held-out accuracy")
    ax[0].set_title(f"{K}-way batch discrimination\nwithin one fixed disease class "
                     f"({TARGET_CLASS.split('_',1)[1]})", pad=12)
    for i, v in enumerate([acc, chance]):
        ax[0].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    # style matched to Fig. 6(b) (confusion_absolute.png): YlOrRd colormap,
    # every cell annotated (including zeros), named axis labels.
    im = ax[1].imshow(cm, cmap="YlOrRd")
    ax[1].set_xlabel("Predicted batch"); ax[1].set_ylabel("True batch")
    ax[1].set_title(f"Batch-ID confusion matrix\n(counts on {int(cm.sum())}-image held-out set)")
    ax[1].set_xticks(range(K)); ax[1].set_yticks(range(K))
    ax[1].set_xticklabels(batch_names, fontsize=8, rotation=45, ha="right")
    ax[1].set_yticklabels(batch_names, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(K):
        for j in range(K):
            ax[1].text(j, i, str(int(cm[i, j])), ha="center", va="center",
                       fontsize=8, color="white" if cm[i, j] > thresh else "black")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "batch_id_diagnostic.png"), dpi=160)
    plt.close()

    # Grad-CAM gallery: what does the batch-ID classifier actually look at?
    cam = GradCAMPlusPlus(model, get_target_layer(model))
    n_show = min(6, K)
    fig, axes = plt.subplots(2, n_show, figsize=(2.6 * n_show, 5.4))
    for i in range(n_show):
        path, _ = [it for it in val_items if it[1] == i][0]
        img = Image.open(path).convert("RGB")
        x = val_ds.transform(img).unsqueeze(0).to(DEVICE)
        heat, pred_c, conf = cam.generate(x)
        overlay = overlay_heatmap(img, heat)
        axes[0, i].imshow(img.resize((224, 224))); axes[0, i].axis("off")
        axes[0, i].set_title(f"batch {batch_names[i]}", fontsize=9)
        axes[1, i].imshow(overlay); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM++\n(batch-ID model)", fontsize=9)
    plt.suptitle(f"What the batch-ID shortcut looks like ({TARGET_CLASS.split('_',1)[1]} only, "
                 f"one image per batch)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "batch_id_gradcam_gallery.png"), dpi=160)
    plt.close()
    cam.remove_hooks()

    result = {
        "target_class": TARGET_CLASS,
        "n_batches": K,
        "batch_names": batch_names,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "accuracy": acc,
        "chance_accuracy": chance,
        "accuracy_over_chance": acc / chance,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(OUT_DIR, "batch_id_diagnostic.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[batch-id] wrote {OUT_DIR}/batch_id_diagnostic.json, "
          f"{FIG_DIR}/batch_id_diagnostic.png, {FIG_DIR}/batch_id_gradcam_gallery.png "
          f"(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
