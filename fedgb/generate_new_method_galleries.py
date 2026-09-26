"""
generate_new_method_galleries.py
----------------------------------
Small qualitative heatmap galleries for Eigen-CAM and Score-CAM, matching
the visual style of the existing Grad-CAM/Grad-CAM++ galleries in
outputs/gradcam/. Grad-CAM, Grad-CAM++, and Saliency already have full
per-class galleries and local-vs-global comparison figures in the paper;
Eigen-CAM and Score-CAM so far only have the quantitative Table VI
overlap numbers. This adds one representative example per method so the
new results have visual grounding too.

For each of {eigencam, scorecam}:
  - one global-model heatmap on a Malignant-class validation image
    -> outputs/<method>/global_malignant_01.png
  - one local-vs-global comparison on the same image, using the H_C
    (oncology-centre) client model, which XAI Finding (6) already singles
    out as the strongest Malignant-class local attention
    -> outputs/<method>/comparison_client_2/compare_malignant_00.png

Run from fedgb/:
    python generate_new_method_galleries.py
"""
import os

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from dataset import CLASS_NAMES, GBCUDataset, get_transforms, IDX_TO_CLASS
from gradcam import EigenCAM, ScoreCAM, overlay_heatmap, save_gradcam_figure, DEVICE
from model import build_model, get_target_layer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
GLOBAL_CKPT = os.path.join(REPO_ROOT, "outputs", "checkpoints", "round_010.pt")
CLIENT2_CKPT = os.path.join(REPO_ROOT, "outputs", "seeds", "seed_42", "local_only_client2.pt")
DATA_DIR = os.path.join(REPO_ROOT, "data", "data", "validation")

INV_NORMALIZE = transforms.Compose([
    transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
])

METHOD_CLASSES = {"eigencam": EigenCAM, "scorecam": ScoreCAM}
METHOD_LABELS = {"eigencam": "Eigen-CAM", "scorecam": "Score-CAM"}


def load_model(ckpt_path):
    model = build_model(pretrained=False).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def make_explainer(method, model):
    target_layer = get_target_layer(model)
    if method == "scorecam":
        return ScoreCAM(model, target_layer, max_channels=64, batch_size=32)
    return EigenCAM(model, target_layer)


def to_original_image(tensor):
    orig = INV_NORMALIZE(tensor).permute(1, 2, 0).numpy()
    orig = np.clip(orig, 0, 1)
    return Image.fromarray((orig * 255).astype(np.uint8))


def main():
    print(f"[info] device = {DEVICE}")
    global_model = load_model(GLOBAL_CKPT)
    client2_model = load_model(CLIENT2_CKPT)

    dataset = GBCUDataset(DATA_DIR, transform=get_transforms(train=False))
    malg_idx = CLASS_NAMES.index("malg")
    sample_idx = next(i for i in range(len(dataset)) if dataset[i][1] == malg_idx)
    tensor, true_label = dataset[sample_idx]
    original_image = to_original_image(tensor)
    input_tensor = tensor.unsqueeze(0)

    for method in ["eigencam", "scorecam"]:
        label = METHOD_LABELS[method]
        out_dir = os.path.join(REPO_ROOT, "outputs", method)
        os.makedirs(out_dir, exist_ok=True)

        # 1) Global-model heatmap, 3-panel figure
        gcam_global = make_explainer(method, global_model)
        heatmap_g, pred_g, conf_g = gcam_global.generate(input_tensor)
        save_gradcam_figure(
            original_image=original_image,
            heatmap=heatmap_g,
            pred_class=pred_g,
            confidence=conf_g,
            true_class=true_label,
            save_path=os.path.join(out_dir, "global_malignant_01.png"),
            title_prefix=f"[{label}, global] ",
            heatmap_label=f"{label} Heatmap",
        )
        gcam_global.remove_hooks()

        # 2) Local (H_C / client_2) vs global comparison, 3-panel figure
        gcam_local = make_explainer(method, client2_model)
        heatmap_l, pred_l, conf_l = gcam_local.generate(input_tensor)
        gcam_local.remove_hooks()

        overlay_local = overlay_heatmap(original_image, heatmap_l)
        overlay_global = overlay_heatmap(original_image, heatmap_g)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(original_image.resize((224, 224)))
        axes[0].set_title(f"Original | True: {IDX_TO_CLASS[true_label]}")
        axes[0].axis("off")
        axes[1].imshow(overlay_local)
        axes[1].set_title(f"Local Model (Client 2 / $H_C$)\nPred: {IDX_TO_CLASS[pred_l]} ({conf_l:.2%})")
        axes[1].axis("off")
        axes[2].imshow(overlay_global)
        axes[2].set_title(f"Global Federated Model\nPred: {IDX_TO_CLASS[pred_g]} ({conf_g:.2%})")
        axes[2].axis("off")
        plt.suptitle(f"{label}: Local vs Global — Client 2 ($H_C$), Malignant", fontsize=13, fontweight="bold")
        plt.tight_layout()

        cmp_dir = os.path.join(out_dir, "comparison_client_2")
        os.makedirs(cmp_dir, exist_ok=True)
        cmp_path = os.path.join(cmp_dir, "compare_malignant_00.png")
        plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[{label}] Saved -> {cmp_path}")

    print("[done] Eigen-CAM / Score-CAM galleries generated.")


if __name__ == "__main__":
    main()
