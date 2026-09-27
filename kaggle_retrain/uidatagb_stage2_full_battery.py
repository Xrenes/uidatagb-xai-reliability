"""
uidatagb_stage2_full_battery.py
=================================
Closes the paper's biggest remaining scientific gap: the five-method XAI
reliability battery (consistency, faithfulness, stability, sanity check,
calibration, shortcut audit) was run on the Stage 1 (leakage-inflated)
classifier, and only Grad-CAM's consistency score was ever re-verified
on the corrected Stage 2 model. This script re-runs the FULL battery,
all six analyses, on the Stage 2 primary and replicate checkpoints, so
every number in the paper's XAI comparison is about the same, honestly-
evaluated classifier its accuracy claims are about.

WHAT this script does, in order:
  1. Materialises the Stage 2 test split (from split_70_20_10.json) into
     the folder-per-class structure fedgb/dataset.py's GBCUDataset
     expects (fedgb/_p0common.py's val_dataset() reads a folder tree,
     not a flat file list, so the JSON split has to be turned into
     symlinks/copies under a class-subfolder layout first).
  2. Runs all six existing fedgb/ reliability-battery scripts UNCHANGED
     -- compute_consistency.py, compute_faithfulness.py,
     compute_stability.py, sanity_checks.py, compute_calibration.py,
     shortcut_audit.py -- by pointing their existing P0_CKPT/P0_VAL_DIR/
     P0_OUT_DIR/P0_FIG_DIR/P0_REPLICATE_CKPTS environment variables (already
     built into fedgb/_p0common.py and compute_consistency.py for
     exactly this purpose) at the Stage 2 checkpoints and materialised
     test split, instead of modifying the scripts themselves.
  3. Splits the six analyses across BOTH T4 GPUs to use the Kaggle
     T4 x2 instance fully: three analyses run pinned to cuda:0, three to
     cuda:1, as two parallel subprocesses, rather than one GPU idling
     while the other does all the work.

WHY reuse the exact same scripts unmodified rather than reimplementing
the battery here: every number already in the paper's Section IV was
produced by these exact scripts on Stage 1 data. Re-running the
identical code path on Stage 2 data is what makes the two sets of
numbers a fair, apples-to-apples comparison -- reimplementing the
methods separately here would reopen exactly the kind of "is this
really the same measurement" question this script exists to close.

HOW TO RUN THIS ON KAGGLE
--------------------------------------------------------------------
1. Datasets needed (reuse what earlier Kaggle sessions already used):
   - "uidatagb" (images + dihedral cluster manifest) -- same as before.
   - "uidatagb-stage2-checkpoints" -- needs to be EXPANDED this time to
     include not just primary_seed42/best.pt but also the three
     replicate checkpoints (replicate_seed7/best.pt,
     replicate_seed123/best.pt, replicate_seed2027/best.pt) and the
     split_70_20_10.json file, all from the uidatagb_retrain_outputs.zip
     you already downloaded locally. Zip up
     primary_seed42/, replicate_seed7/, replicate_seed123/,
     replicate_seed2027/, and split_70_20_10.json together and
     re-upload as a new version of that dataset (or a new dataset --
     either works, just update CKPT_ROOT below to match).
2. Run the path-check pattern used before (a quick os.walk cell) to
   confirm the exact mount path, then update CKPT_ROOT and DATA_ROOT
   below to match -- do not guess, this has burned time twice already.
3. Turn on GPU T4 x2.
4. Run all. This is heavier than the Stage 2 retrain's follow-up-review
   script but lighter than the original 4-model retrain: six analyses,
   each doing inference-only forward/backward passes (no training) on
   72-216-image stratified subsets, split across two GPUs. Expect
   30-60 minutes total.
5. At the end, the script auto-zips its output folder, same as before.
   Use the Output side panel to download, not any printed link.
"""
import json
import os
import shutil
import subprocess
import sys
import time

# =====================================================================
# SECTION 0: Configuration -- CHECK THESE PATHS BEFORE RUNNING
# =====================================================================

DATA_ROOT = "/kaggle/input/datasets/sayed227/uidatagb/uidatagb-corrected/uidatagb"
CLUSTER_MANIFEST_PATH = "/kaggle/input/datasets/sayed227/uidatagb/uidatagb-cluster-manifest.json"

# Real Kaggle slug confirmed via os.listdir("/kaggle/input/datasets/sayed227")
# -- the dataset's display title "UIDATA GB Retrain Outputs" slugifies to
# "uidatagb-retrain-outputs", NOT "uidatagb-stage2-checkpoints" as assumed
# earlier. Do not guess this again -- verify with os.listdir if it changes.
CKPT_ROOT = "/kaggle/input/datasets/sayed227/uidatagb-retrain-outputs"
SPLIT_JSON_PATH = os.path.join(CKPT_ROOT, "split_70_20_10.json")
PRIMARY_CKPT = os.path.join(CKPT_ROOT, "primary_seed42", "best.pt")
REPLICATE_CKPTS = [
    os.path.join(CKPT_ROOT, "replicate_seed7", "best.pt"),
    os.path.join(CKPT_ROOT, "replicate_seed123", "best.pt"),
    os.path.join(CKPT_ROOT, "replicate_seed2027", "best.pt"),
]

FEDGB_DIR = "/kaggle/working/fedgb"  # this script expects fedgb/ uploaded alongside it,
                                      # or copied in via a "fedgb-code" Kaggle Dataset --
                                      # see Step 0 below if fedgb/ is not already present.
MATERIALIZED_TEST_DIR = "/kaggle/working/stage2_test_materialized"
OUT_ROOT = "/kaggle/working/uidatagb_stage2_full_battery_outputs"
os.makedirs(OUT_ROOT, exist_ok=True)

CLASS_NAMES = [
    "01_gallstones",
    "02_abdomen_and_retroperitoneum",
    "03_cholecystitis",
    "04_membranous_and_gangrenous_cholecystitis",
    "05_perforation",
    "06_polyps_and_cholesterol_crystals",
    "07_adenomyomatosis",
    "08_carcinoma",
    "09_various_causes_of_gallbladder_wall_thickening",
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT_ROOT, "run.log"), "a") as f:
        f.write(line + "\n")


# =====================================================================
# SECTION 1: materialise the Stage 2 test split into a class-subfolder
# tree, since fedgb/_p0common.py's val_dataset() reads
# DATA_ROOT/validation/<class>/*.jpg, not a flat JSON file list.
# =====================================================================

def materialize_test_split():
    if os.path.isdir(MATERIALIZED_TEST_DIR) and any(os.scandir(MATERIALIZED_TEST_DIR)):
        log(f"[materialize] {MATERIALIZED_TEST_DIR} already populated, skipping")
        return

    if not os.path.exists(SPLIT_JSON_PATH):
        log(f"FATAL: {SPLIT_JSON_PATH} not found. The checkpoint dataset "
            f"must be expanded to include split_70_20_10.json alongside "
            f"the four model checkpoints -- see this script's docstring.")
        sys.exit(1)

    with open(SPLIT_JSON_PATH) as f:
        split = json.load(f)
    test_items = split["test"]
    log(f"[materialize] {len(test_items)} test images to materialise")

    for cname in CLASS_NAMES:
        os.makedirs(os.path.join(MATERIALIZED_TEST_DIR, cname), exist_ok=True)

    # The split JSON stores the ORIGINAL Kaggle mount path from the run
    # that produced it, which may not match this session's mount path
    # exactly (dataset version suffixes, etc.) -- rebase by matching on
    # the "training/<class>/<filename>" or "validation/<class>/<filename>"
    # suffix instead of trusting the absolute path stored in the JSON.
    def rebase(stored_path):
        norm = stored_path.replace("\\", "/")
        parts = norm.split("/")
        for i, p in enumerate(parts):
            if p.lower() in ("training", "validation"):
                # DATA_ROOT/<training-or-validation>/<class>/<file>
                return os.path.join(DATA_ROOT, *parts[i:])
        return None

    n_copied, n_missing = 0, 0
    for stored_path, label_idx in test_items:
        real_path = rebase(stored_path)
        cname = CLASS_NAMES[label_idx]
        if real_path is None or not os.path.exists(real_path):
            n_missing += 1
            continue
        dest = os.path.join(MATERIALIZED_TEST_DIR, cname, os.path.basename(real_path))
        if not os.path.exists(dest):
            shutil.copy2(real_path, dest)
        n_copied += 1

    log(f"[materialize] copied {n_copied} images, {n_missing} missing "
        f"(missing should be 0 -- investigate DATA_ROOT/rebase logic if not)")


# =====================================================================
# SECTION 2: run the six existing fedgb/ battery scripts, split across
# both GPUs as two parallel subprocess groups.
# =====================================================================

# Group A -> cuda:0 (consistency needs all 4 checkpoints loaded at once,
# so it is the heaviest single job -- give it its own GPU).
GROUP_A = ["compute_consistency.py"]
# Group B -> cuda:1 (the remaining five, run sequentially on the second
# GPU; each only needs the primary checkpoint).
GROUP_B = ["compute_faithfulness.py", "compute_stability.py",
           "sanity_checks.py", "compute_calibration.py", "shortcut_audit.py"]


def run_script(script_name, gpu_id, out_subdir, extra_env=None):
    result_marker = os.path.join(OUT_ROOT, out_subdir, "phase0")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["P0_CKPT"] = PRIMARY_CKPT
    env["P0_VAL_DIR"] = MATERIALIZED_TEST_DIR
    env["P0_OUT_DIR"] = os.path.join(OUT_ROOT, out_subdir, "phase0")
    env["P0_FIG_DIR"] = os.path.join(OUT_ROOT, out_subdir, "figures")
    if extra_env:
        env.update(extra_env)
    os.makedirs(env["P0_OUT_DIR"], exist_ok=True)
    os.makedirs(env["P0_FIG_DIR"], exist_ok=True)

    log(f"[run] starting {script_name} on cuda:{gpu_id}")
    proc = subprocess.run(
        [sys.executable, script_name],
        cwd=FEDGB_DIR, env=env,
        capture_output=True, text=True,
    )
    log(f"[run] {script_name} exit={proc.returncode}")
    if proc.stdout:
        log(f"[run] {script_name} stdout tail:\n" + "\n".join(proc.stdout.splitlines()[-15:]))
    if proc.returncode != 0:
        log(f"[run] {script_name} stderr tail:\n" + "\n".join(proc.stderr.splitlines()[-30:]))
    return proc.returncode == 0


def main():
    log("=== uidatagb_stage2_full_battery.py started ===")

    if not os.path.isdir(DATA_ROOT):
        log(f"FATAL: DATA_ROOT {DATA_ROOT} does not exist.")
        sys.exit(1)
    if not os.path.isdir(FEDGB_DIR):
        log(f"FATAL: {FEDGB_DIR} does not exist. Upload the fedgb/ "
            f"directory's code (not data/outputs, just the .py files) as "
            f"a Kaggle Dataset and copy/symlink it to {FEDGB_DIR} in an "
            f"earlier cell, e.g.:\n"
            f"  !cp -r /kaggle/input/<your-fedgb-code-dataset>/fedgb {FEDGB_DIR}")
        sys.exit(1)
    for p in [PRIMARY_CKPT] + REPLICATE_CKPTS:
        if not os.path.exists(p):
            log(f"FATAL: checkpoint not found at {p}")
            sys.exit(1)

    materialize_test_split()

    # consistency needs the replicate paths passed explicitly
    consistency_env = {"P0_REPLICATE_CKPTS": os.pathsep.join(REPLICATE_CKPTS)}

    # Launch Group A (consistency, cuda:0) and Group B (the rest,
    # cuda:1) as two parallel processes so both GPUs are used at once.
    import concurrent.futures

    def run_group_a():
        return run_script("compute_consistency.py", 0, "consistency",
                           extra_env=consistency_env)

    def run_group_b():
        ok = True
        for script in GROUP_B:
            subdir = script.replace("compute_", "").replace(".py", "")
            ok = run_script(script, 1, subdir) and ok
        return ok

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(run_group_a)
        future_b = executor.submit(run_group_b)
        ok_a = future_a.result()
        ok_b = future_b.result()

    log(f"=== ALL DONE (Group A ok={ok_a}, Group B ok={ok_b}) ===")
    log(f"Outputs written under: {OUT_ROOT}")

    try:
        zip_path = shutil.make_archive(OUT_ROOT, "zip", OUT_ROOT)
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        log(f"[auto-zip] wrote {zip_path} ({size_mb:.1f} MB)")
    except Exception as e:
        log(f"[auto-zip] WARNING: zipping failed ({e})")

    log("TO DOWNLOAD: use the Output side panel, not any printed link "
        "(FileLink-style links 404 on Kaggle).")
    log("IMPORTANT: click 'Save Version -> Save & Run All' before "
        "downloading.")


if __name__ == "__main__":
    main()
