"""
verify_and_prepare_download.py
================================
Run this in a NEW cell, in the SAME Kaggle session, any time after you
see "=== ALL DONE ===" in the main training script's log (or even if
you're not sure it finished -- this only reads files, it never trains
or deletes anything).

WHAT: checks that every expected output file actually exists and looks
      non-empty/well-formed, (re)builds the single zip if needed, and
      prints exactly what to click in the Output panel.
WHY:  a script running inside a Kaggle kernel CANNOT trigger a browser
      download by itself -- Kaggle does not expose that API to notebook
      code, and IPython.display.FileLink 404s on Kaggle because Kaggle
      doesn't serve /kaggle/working over the classic Jupyter file route
      it assumes. The only working download path is Kaggle's own
      "Output" side panel (while the session is alive) or the
      committed version's "Output" tab (after Save Version -> Save &
      Run All). This script's job is to make sure there is something
      complete and correct sitting there ready to click, and to tell
      you precisely where to look -- not to "download" in the sense of
      pushing a file to your browser, which is not possible from here.
OUTPUT: a pass/fail report printed below the cell, and (if missing or
        stale) a fresh uidatagb_retrain_outputs.zip written to
        /kaggle/working/.
"""

import os
import json
import shutil

OUT_ROOT = "/kaggle/working/uidatagb_retrain_outputs"
ZIP_PATH = "/kaggle/working/uidatagb_retrain_outputs.zip"

# Every file the full pipeline is supposed to produce, and a short
# description of why each one matters -- shown next to any that are
# missing so a partial run is easy to diagnose, not just flagged.
EXPECTED_FILES = {
    "split_70_20_10.json": "the cluster-aware 70/20/10 split used by every model",
    "primary_seed42/best.pt": "primary model checkpoint",
    "primary_seed42/history.json": "primary model epoch-by-epoch training curve",
    "primary_seed42/test_metrics.json": "primary model's real held-out test accuracy",
    "replicate_seed7/best.pt": "replicate 1 checkpoint (same data as primary, different seed)",
    "replicate_seed7/test_metrics.json": "replicate 1 test accuracy",
    "replicate_seed123/best.pt": "replicate 2 checkpoint",
    "replicate_seed123/test_metrics.json": "replicate 2 test accuracy",
    "replicate_seed2027/best.pt": "replicate 3 checkpoint",
    "replicate_seed2027/test_metrics.json": "replicate 3 test accuracy",
    "null_control_untrained/consistency.json": "Review-2 fix: untrained-model consistency floor",
    "null_control_random_heatmaps/cosine_floor.json": "Review-4 fix: random-heatmap cosine-similarity floor",
    "consistency_same_data/consistency.json": "Review-3/6 fix: corrected same-data consistency + std",
}


def check_file(rel_path):
    full_path = os.path.join(OUT_ROOT, rel_path)
    if not os.path.exists(full_path):
        return False, "MISSING"
    size = os.path.getsize(full_path)
    if size == 0:
        return False, "EXISTS BUT EMPTY (0 bytes)"
    if rel_path.endswith(".json"):
        try:
            with open(full_path) as f:
                json.load(f)
        except Exception as e:
            return False, f"EXISTS BUT INVALID JSON ({e})"
    return True, f"OK ({size / 1024:.1f} KB)"


def main():
    if not os.path.isdir(OUT_ROOT):
        print(f"FATAL: {OUT_ROOT} does not exist at all.")
        print("This means either training has not been run yet in this "
              "session, or the session restarted and lost everything "
              "(this has happened before -- see chat history). "
              "You need to re-run uidatagb_full_retrain.py from the "
              "start in this case.")
        return

    print(f"Checking expected outputs under {OUT_ROOT} ...\n")
    all_ok = True
    for rel_path, why in EXPECTED_FILES.items():
        ok, status = check_file(rel_path)
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] {rel_path:55s} {status}")
        if not ok:
            all_ok = False
            print(f"         -> why this matters: {why}")

    print()
    if all_ok:
        print("All expected files present and readable. Full pipeline "
              "run looks complete.")
    else:
        print("Some files are missing or invalid (see FAIL lines above). "
              "The run may still be in progress, or may have stopped "
              "partway through -- check the main script's log for "
              "errors before assuming this is final.")

    # Rebuild the zip fresh regardless, so it reflects whatever is
    # actually present right now (useful even on a partial run, so you
    # don't lose the models that DID finish if something later failed).
    print(f"\n(Re)building {ZIP_PATH} from current contents ...")
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    zip_path = shutil.make_archive(OUT_ROOT, "zip", OUT_ROOT)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Wrote {zip_path} ({size_mb:.1f} MB)\n")

    print("=" * 70)
    print("TO DOWNLOAD (this is the ONLY step a script cannot do for you):")
    print("  1. Look at the right-hand sidebar of this notebook.")
    print("  2. Find the 'Output' section -> expand '/kaggle/working'.")
    print("  3. Find 'uidatagb_retrain_outputs.zip' in that list.")
    print("  4. Click the small download icon next to it.")
    print("  Do NOT click any 'FileLink'-style link printed by other ")
    print("  cells -- those 404 on Kaggle regardless of whether the ")
    print("  file exists.")
    print("=" * 70)


main()
