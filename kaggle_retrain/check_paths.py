"""
check_paths.py
--------------
Run this FIRST in the Kaggle notebook, after attaching all three datasets:
  - uidatagb
  - uidatagb-stage2-checkpoints
  - fedgb-code

Paste this whole file into a notebook cell and run it. It walks /kaggle/input
and prints every file so we confirm real mount paths instead of guessing them
(Kaggle nests datasets under /kaggle/input/datasets/<user>/<slug>/... rather
than the /kaggle/input/<slug>/... shown in the "Add data" UI, and this has
been wrong before).
"""
import os

for root, dirs, files in os.walk("/kaggle/input"):
    # skip descending into huge image folders, just show first few + count
    if "uidatagb" in root and "uidatagb-corrected" in root and files:
        print(f"{root}/  ({len(files)} files, showing 3)")
        for f in files[:3]:
            print(f"    {f}")
        dirs[:] = []  # don't recurse further into this image tree
        continue
    for f in files:
        print(os.path.join(root, f))

print("\n--- summary: top-level dataset folders ---")
base = "/kaggle/input"
for d in sorted(os.listdir(base)):
    print(d)
    sub = os.path.join(base, d)
    if os.path.isdir(sub):
        for d2 in sorted(os.listdir(sub)):
            print("  ", d2)
