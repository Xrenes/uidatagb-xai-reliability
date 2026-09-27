"""
uidatagb_group_aware_xai_bootstrap.py
=======================================
Closes audit item C4: XAI confidence intervals (Tables V, VII, VIII) used
plain image-level bootstrap, not the group-aware (dihedral-cluster)
bootstrap already applied to classification-accuracy CIs in Results IV.1.
Correlated near-duplicate frames within a cluster can make an image-level
bootstrap understate true uncertainty for XAI metrics the same way it did
for accuracy.

WHAT this script does: runs the three now-modified fedgb/ scripts
(compute_consistency.py, compute_faithfulness.py, compute_stability.py)
on the STAGE 1 checkpoint and validation set -- the same data these
scripts' outputs already back in the paper's Tables V, VII, VIII -- with
the dihedral cluster manifest supplied via P0_CLUSTER_MANIFEST, so each
script's newly-added group-aware bootstrap actually runs instead of
silently skipping.

Each script now writes BOTH the original image-level bootstrap CI and a
new group-level one (plus the width ratio between them) into the same
JSON output paths as before, so this is a superset of the existing
outputs, not a separate result set.

HOW TO RUN THIS ON KAGGLE
--------------------------------------------------------------------
1. Attach the "uidatagb" dataset (images + cluster manifest) -- same as
   every other Kaggle run this session.
2. NEW dataset needed: "stage1-checkpoints", from
   stage1-checkpoints.zip (contains seed_42/centralized.pt, the Stage 1
   primary checkpoint these tables' numbers actually come from, plus
   seed_42/local_only_client{0,1,2}.pt, the three Stage 1 replicate
   checkpoints compute_consistency.py needs). Upload it as a new
   dataset before running this script.
3. This script downloads the three modified fedgb/ scripts (plus their
   dependencies) directly from GitHub via wget, so no separate
   "fedgb-code" dataset re-upload is needed if it's already stale --
   though attaching fedgb-code-v2 as before also works if you prefer to
   avoid the network fetch; the script checks both.
4. Turn on GPU (a single T4 is enough; these are lightweight,
   inference-only stratified-sample runs, each ~2 minutes, like the
   earlier full-battery run).
5. Run all. Expect under 10 minutes total for all three scripts.
6. Auto-zips output at the end; download via the Output side panel.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

DATA_ROOT = "/kaggle/input/datasets/sayed227/uidatagb/uidatagb-corrected/uidatagb"
CLUSTER_MANIFEST_PATH_INPUT = "/kaggle/input/datasets/sayed227/uidatagb/uidatagb-cluster-manifest.json"

FEDGB_DIR = "/kaggle/working/fedgb"
OUT_ROOT = "/kaggle/working/uidatagb_group_aware_xai_bootstrap_outputs"
os.makedirs(OUT_ROOT, exist_ok=True)

GITHUB_RAW = "https://raw.githubusercontent.com/Xrenes/uidatagb-xai-reliability/main/fedgb"
SCRIPTS_TO_RUN = ["compute_consistency.py", "compute_faithfulness.py", "compute_stability.py"]
DEPENDENCIES = ["_p0common.py", "dataset.py", "model.py", "gradcam.py"]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT_ROOT, "run.log"), "a") as f:
        f.write(line + "\n")


def ensure_fedgb_code():
    """Prefer an already-attached fedgb-code(-v2) dataset if present
    (avoids a network dependency); otherwise fetch the needed files
    directly from GitHub via wget, since the group-aware bootstrap
    changes are already pushed there."""
    for slug in ("fedgb-code-v2", "fedgb-code"):
        candidate = f"/kaggle/input/datasets/sayed227/{slug}/fedgb"
        if os.path.isdir(candidate):
            log(f"[setup] found attached dataset at {candidate}, copying")
            shutil.copytree(candidate, FEDGB_DIR, dirs_exist_ok=True)
            return
    log("[setup] no fedgb-code dataset attached; fetching updated scripts "
        "directly from GitHub instead")
    os.makedirs(FEDGB_DIR, exist_ok=True)
    for fname in SCRIPTS_TO_RUN + DEPENDENCIES:
        url = f"{GITHUB_RAW}/{fname}"
        dest = os.path.join(FEDGB_DIR, fname)
        urllib.request.urlretrieve(url, dest)
        log(f"[setup] fetched {fname} ({os.path.getsize(dest)} bytes)")


def ensure_cluster_manifest():
    """The scripts look for outputs/phase0/dihedral_cluster_manifest.json
    relative to fedgb/ by default; point P0_CLUSTER_MANIFEST at the
    attached dataset's copy instead of duplicating the 1.7MB file."""
    if os.path.exists(CLUSTER_MANIFEST_PATH_INPUT):
        log(f"[setup] using cluster manifest from attached dataset: "
            f"{CLUSTER_MANIFEST_PATH_INPUT}")
        return CLUSTER_MANIFEST_PATH_INPUT
    # fallback: fetch from GitHub if the attached dataset's copy isn't found
    url = f"{GITHUB_RAW}/outputs/phase0/dihedral_cluster_manifest.json"
    dest = os.path.join(FEDGB_DIR, "outputs", "phase0", "dihedral_cluster_manifest.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    log(f"[setup] cluster manifest not found at expected input path, "
        f"fetching from GitHub instead")
    urllib.request.urlretrieve(url, dest)
    log(f"[setup] fetched cluster manifest ({os.path.getsize(dest)} bytes)")
    return dest


def run_script(script_name, ckpt_path, val_dir, cluster_manifest_path):
    out_subdir = script_name.replace("compute_", "").replace(".py", "")
    phase0_dir = os.path.join(OUT_ROOT, out_subdir, "phase0")
    fig_dir = os.path.join(OUT_ROOT, out_subdir, "figures")
    os.makedirs(phase0_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    env = os.environ.copy()
    env["P0_CKPT"] = ckpt_path
    env["P0_VAL_DIR"] = val_dir
    env["P0_OUT_DIR"] = phase0_dir
    env["P0_FIG_DIR"] = fig_dir
    env["P0_CLUSTER_MANIFEST"] = cluster_manifest_path

    log(f"[run] starting {script_name}")
    proc = subprocess.run(
        [sys.executable, script_name],
        cwd=FEDGB_DIR, env=env, capture_output=True, text=True,
    )
    log(f"[run] {script_name} exit={proc.returncode}")
    if proc.stdout:
        log(f"[run] {script_name} stdout tail:\n" + "\n".join(proc.stdout.splitlines()[-25:]))
    if proc.returncode != 0:
        log(f"[run] {script_name} stderr tail:\n" + "\n".join(proc.stderr.splitlines()[-30:]))
    return proc.returncode == 0


def main():
    log("=== uidatagb_group_aware_xai_bootstrap.py started ===")

    if not os.path.isdir(DATA_ROOT):
        log(f"FATAL: DATA_ROOT {DATA_ROOT} does not exist.")
        sys.exit(1)

    ensure_fedgb_code()
    cluster_manifest_path = ensure_cluster_manifest()

    # This audit item concerns the EXISTING Stage 1 tables (V, VII, VIII),
    # which report on the Stage 1 (single-orientation-hash split) primary
    # model -- confirmed to be outputs/seeds/seed_42/centralized.pt by
    # reading fedgb/_p0common.py's own CKPT default directly, not guessed.
    # This checkpoint (plus the 3 Stage 1 replicates compute_consistency.py
    # needs) must be uploaded as a new "stage1-checkpoints" Kaggle dataset
    # from stage1-checkpoints.zip before running this script -- see the
    # docstring above.
    STAGE1_CKPT_ROOT = "/kaggle/input/datasets/sayed227/stage1-checkpoints/seed_42"
    stage1_ckpt = os.path.join(STAGE1_CKPT_ROOT, "centralized.pt")
    stage1_replicates = [
        os.path.join(STAGE1_CKPT_ROOT, f"local_only_client{c}.pt") for c in range(3)
    ]
    if not os.path.exists(stage1_ckpt):
        log(f"FATAL: Stage 1 primary checkpoint not found at {stage1_ckpt}. "
            f"Upload stage1-checkpoints.zip as a new Kaggle dataset named "
            f"'stage1-checkpoints' (see this script's docstring), verify "
            f"the real mount path with an os.walk cell if it differs from "
            f"the above, and update STAGE1_CKPT_ROOT if needed.")
        sys.exit(1)
    for p in stage1_replicates:
        if not os.path.exists(p):
            log(f"FATAL: Stage 1 replicate checkpoint not found at {p}.")
            sys.exit(1)
    log(f"[setup] Stage 1 checkpoints confirmed present: primary + "
        f"{len(stage1_replicates)} replicates")

    val_dir = os.path.join(DATA_ROOT, "validation")

    # compute_consistency.py additionally needs P0_REPLICATE_CKPTS
    os.environ["P0_REPLICATE_CKPTS"] = os.pathsep.join(stage1_replicates)

    results_ok = {}
    for script in SCRIPTS_TO_RUN:
        results_ok[script] = run_script(
            script,
            ckpt_path=stage1_ckpt,
            val_dir=val_dir,
            cluster_manifest_path=cluster_manifest_path,
        )

    log(f"=== ALL DONE: {results_ok} ===")

    try:
        zip_path = shutil.make_archive(OUT_ROOT, "zip", OUT_ROOT)
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        log(f"[auto-zip] wrote {zip_path} ({size_mb:.1f} MB)")
    except Exception as e:
        log(f"[auto-zip] WARNING: zipping failed ({e})")

    log("TO DOWNLOAD: use the Output side panel, not any printed link.")
    log("IMPORTANT: click 'Save Version -> Save & Run All' before downloading.")


if __name__ == "__main__":
    main()
