"""
group_aware_statistics.py
---------------------------
Phase 5 of the revision plan: images within a filename-prefix group are
correlated, so a plain image-level bootstrap (as in
stage1_statistical_tests.py) understates uncertainty. This recomputes
confidence intervals using a GROUP bootstrap (resample whole filename-prefix
groups with replacement, keep every image in a resampled group) on the
Stage-1 (pHash-corrected, still group-confounded) validation set, for the
three existing seeds (7, 42, 123).

Also adds the two metrics the plan flags as missing from Table 3 throughout:
macro PR-AUC (average precision, one-vs-rest) and balanced accuracy.

This does NOT change any headline number in the manuscript -- it is a
supplementary uncertainty-quantification check showing how much wider the
group-aware CI is than the naive per-image CI already reported in
outputs/phase0/stage1_statistical_tests.json.

Run from fedgb/:  python group_aware_statistics.py
Outputs: outputs/phase0/group_aware_statistics.json
"""
import json
import os
import re

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                               average_precision_score, balanced_accuracy_score)

from dataset import CLASS_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [7, 42, 123]
DATA_DIR = "./data/uidatagb"
N_BOOT = 2000
RNG_SEED = 42
FNAME_RE = re.compile(r'^([A-Za-z]+\d+)\s*\(\d+\)\.\w+$')


def session_key(fname):
    m = FNAME_RE.match(fname)
    return m.group(1).lower() if m else f"__unmatched__{fname}"


def build_group_ids_for_validation():
    """Reconstructs the (image_index -> filename-prefix-group) mapping in the
    same order GBCUDataset builds self.samples (iterate CLASS_NAMES, then
    os.listdir per class dir) -- the same order used to produce y_true/
    y_pred/y_probs when the seed JSONs were written by run/eval scripts."""
    group_ids = []
    val_dir = os.path.join(DATA_DIR, "validation")
    for cname in CLASS_NAMES:
        cdir = os.path.join(val_dir, cname)
        if not os.path.isdir(cdir):
            continue
        for fname in os.listdir(cdir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                group_ids.append(session_key(fname))
    return group_ids


def load_seed(seed):
    path = os.path.join(HERE, "outputs", "seeds", f"seed_{seed}", "centralized.json")
    with open(path) as f:
        d = json.load(f)
    return np.array(d["y_true"]), np.array(d["y_pred"]), np.array(d["y_probs"])


def extra_metrics(y_true, y_pred, y_probs, n_classes):
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    y_bin = np.eye(n_classes)[y_true]
    try:
        pr_auc = average_precision_score(y_bin, y_probs, average="macro")
    except ValueError:
        pr_auc = float("nan")
    return bal_acc, pr_auc


def group_bootstrap_ci(y_true, y_pred, y_probs, group_ids, n_classes, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    group_keys = list(groups.keys())
    n_groups = len(group_keys)

    accs, f1s, aucs, praucs, balaccs = [], [], [], [], []
    for _ in range(n_boot):
        sampled_groups = rng.choice(group_keys, size=n_groups, replace=True)
        idx = np.concatenate([groups[g] for g in sampled_groups])
        yt, yp, ypr = y_true[idx], y_pred[idx], y_probs[idx]
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        balaccs.append(balanced_accuracy_score(yt, yp))
        try:
            y_bin = np.eye(n_classes)[yt]
            aucs.append(roc_auc_score(y_bin, ypr, average="macro", multi_class="ovr"))
            praucs.append(average_precision_score(y_bin, ypr, average="macro"))
        except ValueError:
            pass  # resample missing a class

    def ci(vals):
        vals = np.sort(np.array(vals))
        return (float(np.mean(vals)), float(vals[int(0.025 * len(vals))]),
                float(vals[int(0.975 * len(vals)) - 1]))

    return {
        "n_groups": n_groups,
        "accuracy": ci(accs),
        "f1_macro": ci(f1s),
        "auc_macro": ci(aucs),
        "pr_auc_macro": ci(praucs),
        "balanced_accuracy": ci(balaccs),
    }


def image_level_bootstrap_ci(y_true, y_pred, y_probs, n_classes, n_boot=N_BOOT, seed=RNG_SEED):
    """Same as stage1_statistical_tests.py's bootstrap, but also reports
    PR-AUC and balanced accuracy, for direct width comparison against the
    group-aware CI above."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s, aucs, praucs, balaccs = [], [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_probs[idx]
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        balaccs.append(balanced_accuracy_score(yt, yp))
        try:
            y_bin = np.eye(n_classes)[yt]
            aucs.append(roc_auc_score(y_bin, ypr, average="macro", multi_class="ovr"))
            praucs.append(average_precision_score(y_bin, ypr, average="macro"))
        except ValueError:
            pass

    def ci(vals):
        vals = np.sort(np.array(vals))
        return (float(np.mean(vals)), float(vals[int(0.025 * len(vals))]),
                float(vals[int(0.975 * len(vals)) - 1]))

    return {"accuracy": ci(accs), "f1_macro": ci(f1s), "auc_macro": ci(aucs),
            "pr_auc_macro": ci(praucs), "balanced_accuracy": ci(balaccs)}


def main():
    group_ids = build_group_ids_for_validation()
    print(f"[group-stats] reconstructed {len(group_ids)} validation group IDs "
          f"({len(set(group_ids))} distinct filename-prefix groups)")

    results = {}
    for s in SEEDS:
        y_true, y_pred, y_probs = load_seed(s)
        assert len(y_true) == len(group_ids), (
            f"seed {s}: y_true has {len(y_true)} entries but reconstructed "
            f"{len(group_ids)} group IDs -- validation directory may have "
            f"changed since this seed was evaluated"
        )
        n_classes = int(y_true.max()) + 1

        bal_acc, pr_auc = extra_metrics(y_true, y_pred, y_probs, n_classes)
        print(f"[group-stats] seed={s}: balanced_accuracy={bal_acc:.4f}  "
              f"macro_pr_auc={pr_auc:.4f}")

        img_ci = image_level_bootstrap_ci(y_true, y_pred, y_probs, n_classes)
        grp_ci = group_bootstrap_ci(y_true, y_pred, y_probs, group_ids, n_classes)

        print(f"[group-stats] seed={s} accuracy CI width: "
              f"image-level={img_ci['accuracy'][2]-img_ci['accuracy'][1]:.4f}  "
              f"group-aware={grp_ci['accuracy'][2]-grp_ci['accuracy'][1]:.4f} "
              f"(n_groups={grp_ci['n_groups']})")

        results[str(s)] = {
            "point_estimates": {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "balanced_accuracy": float(bal_acc),
                "pr_auc_macro": float(pr_auc),
            },
            "image_level_bootstrap_ci": img_ci,
            "group_aware_bootstrap_ci": grp_ci,
        }

    out_dir = os.path.join(HERE, "outputs", "phase0")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "group_aware_statistics.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_val_images": len(group_ids),
            "n_distinct_groups": len(set(group_ids)),
            "note": ("Group bootstrap resamples whole filename-prefix groups with "
                      "replacement (not individual images), matching Phase 5's "
                      "requirement not to treat within-group images as independent "
                      "observations. This is computed on the Stage-1 (pHash-corrected "
                      "but still group-confounded) split; group identity is not "
                      "verified as true patient/session identity (Section 3.1)."),
            "per_seed": results,
        }, f, indent=2)
    print(f"\n[group-stats] wrote {out_path}")


if __name__ == "__main__":
    main()
