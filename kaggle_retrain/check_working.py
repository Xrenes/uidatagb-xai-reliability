"""
check_working.py
==================
Run this in a new cell if zip_and_download.py says the outputs folder
doesn't exist, even though the training log showed "=== ALL DONE ===".

WHAT: lists everything currently under /kaggle/working/ so we can see
      whether the outputs are there under a different name/path, or
      whether the session actually reset and the outputs are gone.
WHY:  Kaggle's /kaggle/working/ is tied to the notebook's running
      kernel. If the kernel restarted (session timeout, "Factory
      reset", or navigating away and the draft session died), anything
      not explicitly saved via "Save Version" is lost, even though the
      log text itself may still be visible in an old cell's output.
"""

import os

ROOT = "/kaggle/working"

if not os.path.isdir(ROOT):
    print(f"FATAL: {ROOT} itself does not exist. This would be very "
          f"unusual for a Kaggle notebook -- something is badly wrong "
          f"with this session.")
else:
    print(f"Contents of {ROOT}:\n")
    for root, dirs, files in os.walk(ROOT):
        depth = root.replace(ROOT, "").count(os.sep)
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or root}/")
        if depth < 4:
            for f in sorted(files)[:5]:
                full = os.path.join(root, f)
                size_kb = os.path.getsize(full) / 1024
                print(f"{indent}  {f}  ({size_kb:.1f} KB)")
            if len(files) > 5:
                print(f"{indent}  ... ({len(files) - 5} more files)")

    print("\nIf uidatagb_retrain_outputs/ is NOT listed above, the "
          "training session's kernel likely restarted and the results "
          "were lost (Kaggle does not persist /kaggle/working across a "
          "kernel restart unless 'Save Version' already ran). In that "
          "case the training needs to be re-run, and 'Save Version -> "
          "Save & Run All' should be clicked immediately once it "
          "finishes, before running anything else in a new cell.")
