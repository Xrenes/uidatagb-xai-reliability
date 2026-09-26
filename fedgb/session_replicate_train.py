"""
session_replicate_train.py
---------------------------
Follow-up to session_leakage_check.py: trains and SAVES checkpoints for the
primary model plus three replicate models on the session/batch-corrected
UIdataGB split (data/uidatagb_sessioncorrected/), so the full XAI
reliability battery (consistency, faithfulness, stability, sanity check,
calibration, shortcut audit) can eventually be recomputed on this split,
matching the methodology already used for the pHash-corrected split.

Run from fedgb/:  python session_replicate_train.py
Outputs: outputs/session_corrected/seed_42/{primary,replicate_1,replicate_2,replicate_3}.pt
         outputs/session_corrected/seed_42/{primary,replicate_1,replicate_2,replicate_3}_metrics.json
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import get_transforms, GBCUDataset
from model import build_model
from evaluate import run_inference, compute_metrics

DATA_DIR = "./data/uidatagb_sessioncorrected"
OUT_DIR = "./outputs/session_corrected/seed_42"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)


def train_one(train_ds, val_ds, tag, epochs=10):
    t0 = time.time()
    torch.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_model(pretrained=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
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
        print(f"[{tag}] epoch {epoch+1}/{epochs} loss={running_loss/len(train_ds):.4f} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    model.eval()
    y_true, y_pred, y_probs = run_inference(model, val_loader)
    metrics = compute_metrics(y_true, y_pred, y_probs)
    print(f"[{tag}] accuracy={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} "
          f"auc={metrics['auc_macro']:.4f}", flush=True)

    ckpt_path = os.path.join(OUT_DIR, f"{tag}.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(OUT_DIR, f"{tag}_metrics.json"), "w") as f:
        json.dump({"accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"],
                    "auc_macro": metrics["auc_macro"], "n_train": len(train_ds),
                    "n_val": len(val_ds)}, f, indent=2)
    print(f"[{tag}] wrote {ckpt_path}", flush=True)
    return metrics


def main():
    full_train = GBCUDataset(os.path.join(DATA_DIR, "training"), transform=get_transforms(train=True))
    val_ds = GBCUDataset(os.path.join(DATA_DIR, "validation"), transform=get_transforms(train=False))

    # primary: full training set
    train_one(full_train, val_ds, "primary")

    # three replicates: disjoint random thirds of the training set (matches
    # the original replicate-subset methodology used for the pHash-corrected
    # split, Section 3.3)
    rng = random.Random(SEED)
    idx = list(range(len(full_train)))
    rng.shuffle(idx)
    thirds = np.array_split(idx, 3)
    for i, sub_idx in enumerate(thirds, start=1):
        sub_ds = Subset(full_train, sub_idx.tolist())
        train_one(sub_ds, val_ds, f"replicate_{i}")

    print("[session_replicate_train] all done.", flush=True)


if __name__ == "__main__":
    main()
