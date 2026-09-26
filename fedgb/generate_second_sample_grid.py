"""
generate_second_sample_grid.py
--------------------------------
Composites the existing per-class Grad-CAM++ gallery's *second* sample
(sample_02, distinct from the sample_01 already used in Fig. 9's
cross-method grid) into one 3x3 main-body figure -- a second, independent
heatmap example per class, promoted from Supplementary Material A into
the main text.

Pure image composition, no model loading / no retraining.

Run from fedgb/:  python generate_second_sample_grid.py
Outputs: outputs/figures/per_class_heatmap_grid.png
"""
import os
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PER_CLASS_DIR = os.path.join(HERE, "outputs", "gradcam_uidatagb", "per_class")
FIG_DIR = os.path.join(HERE, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_ORDER = [
    ("01_gallstones_02.png", "Gallstones (GS)"),
    ("02_abdomen_and_retroperitoneum_02.png", "Abdomen & Retroperitoneum (ART)"),
    ("03_cholecystitis_02.png", "Cholecystitis (CHOL)"),
    ("04_membranous_and_gangrenous_cholecystitis_02.png", "Membranous/Gangrenous\nCholecystitis (MGC)"),
    ("05_perforation_02.png", "Perforation (PERF)"),
    ("06_polyps_and_cholesterol_crystals_02.png", "Polyps & Cholesterol\nCrystals (PCC)"),
    ("07_adenomyomatosis_02.png", "Adenomyomatosis (ADNM)"),
    ("08_carcinoma_02.png", "Carcinoma (CARC)"),
    ("09_various_causes_of_gallbladder_wall_thickening_02.png", "Wall Thickening (WT)"),
]


def main():
    fig, axes = plt.subplots(3, 3, figsize=(9, 9.5))
    for ax, (fname, label) in zip(axes.flat, CLASS_ORDER):
        img = Image.open(os.path.join(PER_CLASS_DIR, fname))
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "per_class_heatmap_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[grid] wrote {out_path}")


if __name__ == "__main__":
    main()
