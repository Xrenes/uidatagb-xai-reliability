"""
anonymize_panels.py
--------------------
One-time fix: the panels built by build_radiologist_study.py were named
things like "nml_01_wrong.png" / "abn_05_correct.png" — the filename itself
spells out whether the model was right, which a rater sees while filling
rating_template.csv. That silently breaks the blinding the study protocol
claims. This renames every panel to a neutral "item_NNN.png", rewrites
manifest.json's "panel" field to match (everything else in manifest.json is
unchanged — it's never shown to raters), and regenerates rating_template.csv.

Idempotent: if panels are already neutral, this is a no-op.

Run from fedgb/:  python anonymize_panels.py
"""
import csv
import json
import os

OUT = "./outputs/radiologist_study"
MANIFEST_PATH = os.path.join(OUT, "manifest.json")
PANELS_DIR = os.path.join(OUT, "panels")


def main():
    manifest = json.load(open(MANIFEST_PATH))
    items = manifest["items"]

    if all(it["panel"].startswith("item_") for it in items):
        print("[skip] panels already anonymized.")
        return

    for it in items:
        old_name = it["panel"]
        new_name = f"item_{it['rating_id']:03d}.png"
        old_path = os.path.join(PANELS_DIR, old_name)
        new_path = os.path.join(PANELS_DIR, new_name)
        if os.path.exists(old_path):
            os.replace(old_path, new_path)
        it["panel"] = new_name

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(OUT, "rating_template.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rating_id", "panel", "plausibility_1to5",
                    "attends_gb_region_yn", "notes"])
        for it in items:
            w.writerow([it["rating_id"], it["panel"], "", "", ""])

    print(f"[done] anonymized {len(items)} panels -> item_NNN.png, "
          f"updated manifest.json and rating_template.csv")


if __name__ == "__main__":
    main()
