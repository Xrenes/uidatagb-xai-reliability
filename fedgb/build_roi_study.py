"""
build_roi_study.py
-------------------
Companion to build_radiologist_study.py / RADIOLOGIST_STUDY.html. That study
shows radiologists a Grad-CAM++ *overlay* and asks for a plausibility rating.
This script prepares the other half: the *raw* (no heatmap) versions of the
same 50 images, so radiologists can draw a gallbladder / pathology bounding
box without the heatmap biasing where they draw it. Drawing a box on top of
the heatmap you're trying to validate would make the IoU/Dice/pointing-game
metrics circular.

Reuses the existing manifest.json (same 50 images, same rating_id) so the
two studies line up image-for-image.

Run from fedgb/:  python build_roi_study.py
Outputs:
  outputs/radiologist_study/panels_raw/<panel>.png   raw image, no overlay
  roi_annotation_tool.html (repo root)                embeds the image list
"""
import json
import os

import numpy as np
from PIL import Image

from dataset import CLASS_NAMES

OUT = "./outputs/radiologist_study"
VAL_DIR = "./data/data/validation"
DISPLAY_SIZE = 512  # upscale for comfortable box-drawing; keep aspect ratio


def main():
    manifest = json.load(open(os.path.join(OUT, "manifest.json")))
    raw_dir = os.path.join(OUT, "panels_raw")
    os.makedirs(raw_dir, exist_ok=True)

    if not manifest["items"][0]["panel"].startswith("item_"):
        raise RuntimeError(
            "Panels are not anonymized yet -- run anonymize_panels.py first "
            "(panel filenames currently leak model correctness).")

    tool_items = []
    for item in manifest["items"]:
        src_path = os.path.join(VAL_DIR, item["true_class"], item["src"])
        img = Image.open(src_path).convert("RGB")
        w, h = img.size
        scale = DISPLAY_SIZE / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
        out_path = os.path.join(raw_dir, item["panel"])
        img.save(out_path)
        tool_items.append({
            "id": item["rating_id"],
            "panel": item["panel"],
            # true_class shown is fine (it's on the request form for the
            # plausibility study too); prediction/correctness stay hidden.
            "true_class": item["true_class"],
        })

    print(f"[done] wrote {len(tool_items)} raw panels to {raw_dir}/")

    # Inject the item list into the annotation tool template so it works
    # from a plain double-clicked HTML file with no server / fetch needed.
    template_path = os.path.join("..", "roi_annotation_tool_template.html")
    out_html_path = os.path.join("..", "roi_annotation_tool.html")
    with open(template_path, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__ITEMS_JSON__", json.dumps(tool_items))
    html = html.replace("__IMG_DIR__",
                        "fedgb/outputs/radiologist_study/panels_raw/")
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[done] wrote {out_html_path}")


if __name__ == "__main__":
    main()
