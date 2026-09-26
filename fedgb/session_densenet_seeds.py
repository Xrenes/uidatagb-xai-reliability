"""
session_densenet_seeds.py
---------------------------
Phase 3.3 second-backbone comparison: trains DenseNet-121 on the same
Stage-2 (session/batch-corrected) split and protocol used for the
ResNet-50 5-seed replication (session_extra_seeds_5.py), at the same 5
seeds. Tests whether the ResNet-50 leakage/generalization conclusion
(accuracy collapsing from ~93% to ~30% under group separation) survives a
different architecture, or is ResNet-50-specific.

Run from fedgb/:  python session_densenet_seeds.py
Outputs: outputs/session_corrected_densenet121/seed_<seed>/primary.pt
         outputs/session_corrected_densenet121/seed_<seed>/primary_metrics.json
         outputs/session_corrected_densenet121/multiseed_summary.json
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
from model import build_model_densenet121
from evaluate import run_inference, compute_metrics

DATA_DIR = "./data/uidatagb_sessioncorrected"
OUT_ROOT = "./outputs/session_corrected_densenet121"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model = build_model_densenet121(pretrained=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    resume_path = os.path.join(out_dir, "resume.pt")
    start_epoch = 0
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1
        t0 -= ckpt["elapsed"]
        print(f"[densenet121 seed={seed}] resuming from epoch {start_epoch+1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
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
        print(f"[densenet121 seed={seed}] epoch {epoch+1}/{epochs} "
              f"loss={running_loss/len(train_ds):.4f} ({time.time()-t0:.0f}s elapsed)", flush=True)
        torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(),
                    "epoch": epoch, "elapsed": time.time() - t0},
                   resume_path)

    if os.path.exists(resume_path):
        os.remove(resume_path)

    model.eval()
    y_true, y_pred, y_probs = run_inference(model, val_loader)
    metrics = compute_metrics(y_true, y_pred, y_probs)
    print(f"[densenet121 seed={seed}] accuracy={metrics['accuracy']:.4f} "
          f"f1={metrics['f1_macro']:.4f} auc={metrics['auc_macro']:.4f}", flush=True)

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
            print(f"[info] loaded existing densenet121 seed={seed} result: acc={m['accuracy']:.4f}", flush=True)
        else:
            results.append(train_one(seed, train_ds, val_ds))

    accs = [r["accuracy"] for r in results]
    f1s = [r["f1_macro"] for r in results]
    aucs = [r["auc_macro"] for r in results]
    summary = {
        "backbone": "densenet121",
        "seeds": [r["seed"] for r in results],
        "n_seeds": len(results),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)), "f1_macro_std": float(np.std(f1s)),
        "auc_macro_mean": float(np.mean(aucs)), "auc_macro_std": float(np.std(aucs)),
        "per_seed": results,
    }
    with open(os.path.join(OUT_ROOT, "multiseed_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[multiseed densenet121] {len(results)}-seed Stage-2 summary: "
          f"accuracy={summary['accuracy_mean']:.4f}+/-{summary['accuracy_std']:.4f}  "
          f"F1={summary['f1_macro_mean']:.4f}+/-{summary['f1_macro_std']:.4f}  "
          f"AUC={summary['auc_macro_mean']:.4f}+/-{summary['auc_macro_std']:.4f}", flush=True)


if __name__ == "__main__":
    main()
