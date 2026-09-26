"""
check_paths.py
===============
Run this FIRST, in its own notebook cell, before uidatagb_full_retrain.py.

WHAT: walks /kaggle/input and prints the full folder tree so the real
      mount path of the attached "uidatagb" dataset can be read off
      directly, instead of guessed.
WHY:  uidatagb_full_retrain.py already failed once with
      "DATA_ROOT ... does not exist" because the dataset's actual
      nested folder structure on Kaggle didn't match what was assumed
      locally. Guessing again risks the same failure and wastes GPU
      session time. This script costs nothing to run and removes the
      guesswork entirely.
OUTPUT: printed tree of every folder (and up to 3 example files per
        folder) under /kaggle/input, to the notebook's output. Copy
        that output back so DATA_ROOT and CLUSTER_MANIFEST_PATH in
        uidatagb_full_retrain.py can be set to match exactly.
"""

import os

ROOT = "/kaggle/input"

if not os.path.isdir(ROOT):
    print(f"FATAL: {ROOT} does not exist. Are you running this inside a "
          f"Kaggle notebook with at least one dataset attached under "
          f"Input? Attach a dataset first, then re-run this cell.")
else:
    print(f"Walking {ROOT} ...\n")
    for root, dirs, files in os.walk(ROOT):
        depth = root.replace(ROOT, "").count(os.sep)
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or root}/")
        if depth < 4:
            for f in sorted(files)[:3]:
                print(f"{indent}  {f}")
            if len(files) > 3:
                print(f"{indent}  ... ({len(files) - 3} more files)")

    print("\nDone. Copy everything above and send it back so DATA_ROOT "
          "and CLUSTER_MANIFEST_PATH can be set correctly in "
          "uidatagb_full_retrain.py before the main run.")
