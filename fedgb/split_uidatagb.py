"""
split_uidatagb.py
------------------
UIdataGB ("Gallblader Diseases Dataset") ships as 9 flat class folders with
no train/validation split. GBCU's split was ~70/30 (1,605/689 of 2,294); we
mirror that ratio here with a stratified 70/30 split per class, seed=42 for
consistency with the primary-model convention used throughout this project.

No patient identifiers are available for this dataset either (same caveat
as GBCU), so this is an image-level stratified split, not a patient-level
one -- disclosed as a limitation in the new paper, same as for GBCU.

Moves (not copies) files into training/ and validation/ subfolders so disk
usage doesn't double; the original flat class folders are removed once
their contents are relocated.

Run from fedgb/:  python split_uidatagb.py
"""
import os
import random

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uidatagb")
TRAIN_FRAC = 0.70
SEED = 42

CLASS_DIRS = [
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


def main():
    rng = random.Random(SEED)
    train_dir = os.path.join(SRC, "training")
    val_dir = os.path.join(SRC, "validation")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    total_train = total_val = 0
    for cname in CLASS_DIRS:
        cpath = os.path.join(SRC, cname)
        if not os.path.isdir(cpath):
            print(f"[skip] {cname} not found (already split?)")
            continue
        files = sorted(f for f in os.listdir(cpath) if f.lower().endswith(".jpg"))
        rng.shuffle(files)
        n_train = round(len(files) * TRAIN_FRAC)

        ctrain = os.path.join(train_dir, cname)
        cval = os.path.join(val_dir, cname)
        os.makedirs(ctrain, exist_ok=True)
        os.makedirs(cval, exist_ok=True)

        for f in files[:n_train]:
            os.replace(os.path.join(cpath, f), os.path.join(ctrain, f))
        for f in files[n_train:]:
            os.replace(os.path.join(cpath, f), os.path.join(cval, f))

        total_train += n_train
        total_val += len(files) - n_train
        print(f"{cname:55s} train={n_train:5d}  val={len(files)-n_train:5d}")

        os.rmdir(cpath)  # now-empty flat folder

    print("---")
    print(f"TOTAL: train={total_train}  val={total_val}  "
          f"({total_train/(total_train+total_val)*100:.1f}% / "
          f"{total_val/(total_train+total_val)*100:.1f}%)")


if __name__ == "__main__":
    main()
