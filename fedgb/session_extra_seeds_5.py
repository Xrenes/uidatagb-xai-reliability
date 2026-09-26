"""
session_extra_seeds_5.py
-------------------------
Extends the Stage-2 (batch-corrected) multi-seed replication from 3 seeds
(7, 42, 123 -- see session_extra_seeds.py) to 5 seeds, per the revision
plan's Phase 3.1 minimum ("at least five same-data seeds"). Trains seeds
2026 and 2027 on the identical Stage-2 split and protocol, then rewrites
multiseed_summary.json with all 5 seeds.

Run from fedgb/:  python session_extra_seeds_5.py
Outputs: outputs/session_corrected/seed_{2026,2027}/primary.pt
         outputs/session_corrected/seed_{2026,2027}/primary_metrics.json
         outputs/session_corrected/multiseed_summary.json (overwritten, now n=5)
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_transforms, GBCUDataset
from model import build_model
from evaluate import run_inference, compute_metrics

DATA_DIR = "./data/uidatagb_sessioncorrected"
OUT_ROOT = "./outputs/session_corrected"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTRA_SEEDS = [2026, 2027]
ALL_SEEDS = [7, 42, 123, 2026, 2027]


def train_one(seed, train_ds, val_ds, epochs=10):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    out_dir = os.path.join(OUT_ROOT, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

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
        print(f"[seed={seed}] epoch {epoch+1}/{epochs} loss={running_loss/len(train_ds):.4f} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    model.eval()
    y_true, y_pred, y_probs = run_inference(model, val_loader)
    metrics = compute_metrics(y_true, y_pred, y_probs)
    print(f"[seed={seed}] accuracy={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} "
          f"auc={metrics['auc_macro']:.4f}", flush=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "primary.pt"))
    result = {"seed": seed, "accuracy": metrics["accuracy"],
              "f1_macro": metrics["f1_macro"], "auc_macro": metrics["auc_macro"]}
    with open(os.path.join(out_dir, "primary_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    train_ds = GBCUDataset(os.path.join(DATA_DIR, "training"), transform=get_transforms(train=True))
    val_ds = GBCUDataset(os.path.join(DATA_DIR, "validation"), transform=get_transforms(train=False))

    results = []
    for seed in ALL_SEEDS:
        metrics_path = os.path.join(OUT_ROOT, f"seed_{seed}", "primary_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            results.append({"seed": seed, "accuracy": m["accuracy"],
                             "f1_macro": m["f1_macro"], "auc_macro": m["auc_macro"]})
            print(f"[info] loaded existing seed={seed} result: acc={m['accuracy']:.4f}", flush=True)
        else:
            results.append(train_one(seed, train_ds, val_ds))

    accs = [r["accuracy"] for r in results]
    f1s = [r["f1_macro"] for r in results]
    aucs = [r["auc_macro"] for r in results]
    summary = {
        "seeds": [r["seed"] for r in results],
        "n_seeds": len(results),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)), "f1_macro_std": float(np.std(f1s)),
        "auc_macro_mean": float(np.mean(aucs)), "auc_macro_std": float(np.std(aucs)),
        "per_seed": results,
    }
    with open(os.path.join(OUT_ROOT, "multiseed_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[multiseed] {len(results)}-seed Stage-2 summary: "
          f"accuracy={summary['accuracy_mean']:.4f}+/-{summary['accuracy_std']:.4f}  "
          f"F1={summary['f1_macro_mean']:.4f}+/-{summary['f1_macro_std']:.4f}  "
          f"AUC={summary['auc_macro_mean']:.4f}+/-{summary['auc_macro_std']:.4f}", flush=True)


if __name__ == "__main__":
    main()
