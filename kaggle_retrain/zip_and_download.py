"""
zip_and_download.py
=====================
Run this in a NEW cell, in the SAME Kaggle notebook that just finished
training, AFTER you see "=== ALL DONE ===" in the log.

WHAT: zips /kaggle/working/uidatagb_retrain_outputs/ into one file and
      creates a direct-download link inside the notebook's own output,
      so you don't have to rely on "Save Version -> Output tab" (which
      also works, but takes longer since it re-runs the whole notebook).
WHY:  Kaggle does not persist /kaggle/working across sessions unless
      you explicitly save it. This gives you an immediate download
      link from the CURRENT running session, without waiting for a
      full "Save & Run All" to complete.
OUTPUT: uidatagb_retrain_outputs.zip in /kaggle/working/, plus a
        clickable download link printed below the cell.
"""

import shutil
import os
from IPython.display import FileLink, display

SRC = "/kaggle/working/uidatagb_retrain_outputs"
ZIP_BASE = "/kaggle/working/uidatagb_retrain_outputs"  # shutil appends .zip

if not os.path.isdir(SRC):
    print(f"FATAL: {SRC} does not exist. Has training finished? "
          f"Check the main script's log for '=== ALL DONE ==='.")
else:
    print(f"Zipping {SRC} ...")
    zip_path = shutil.make_archive(ZIP_BASE, "zip", SRC)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Done: {zip_path} ({size_mb:.1f} MB)\n")
    print("Click the link below to download directly:")
    display(FileLink(zip_path))
