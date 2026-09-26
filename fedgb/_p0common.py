"""
_p0common.py
------------
Shared helpers for the Phase-0 quantitative-XAI validation scripts
(sanity_checks, compute_stability, compute_faithfulness,
compute_calibration, shortcut_audit).

All Phase-0 analyses run on the SAME primary model checkpoint
(seed_42/centralized.pt, UIdataGB, 9 classes) whose accuracy/F1/AUC are
reported in the paper's results tables, so every result is internally
consistent with the model the paper describes.

NOTE: this file previously pointed at the GBCU checkpoint (round_010.pt,
5 classes). That study is archived in full under GBCU_Study/ at the
project root -- see GBCU_Study/fedgb/_p0common.py for the original.
"""
import os
import sys
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import GBCUDataset, get_transforms, CLASS_NAMES, CLASS_LABELS  # noqa: E402
from model import build_model, get_target_layer  # noqa: E402
import evaluate  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluate.DEVICE = DEVICE

# Stage-1 (near-duplicate-corrected) paths are the default, unchanged
# behaviour. Set the P0_* environment variables to point every Phase-0
# script at the Stage-2 (batch-corrected) checkpoints/data/output dirs
# instead, without editing each script individually -- see
# run_stage2_pipeline.py.
CKPT = os.environ.get(
    "P0_CKPT", os.path.join(HERE, "outputs", "seeds", "seed_42", "centralized.pt"))
VAL_DIR = os.environ.get(
    "P0_VAL_DIR", os.path.join(HERE, "data", "uidatagb", "validation"))
OUT_DIR = os.environ.get(
    "P0_OUT_DIR", os.path.join(HERE, "outputs", "phase0"))
FIG_DIR = os.environ.get(
    "P0_FIG_DIR", os.path.join(HERE, "outputs", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CLASS_PRETTY = CLASS_LABELS


def load_primary_model():
    m = build_model(pretrained=False).to(DEVICE)
    m.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    m.eval()
    return m


def val_dataset():
    return GBCUDataset(VAL_DIR, transform=get_transforms(train=False))


def stratified_indices(ds, per_class, seed=42):
    """Return sorted indices with up to `per_class` images from each class."""
    random.seed(seed)
    by = {}
    for i, (_, l) in enumerate(ds.samples):
        by.setdefault(l, []).append(i)
    out = []
    for l, idxs in by.items():
        random.shuffle(idxs)
        out += idxs[:per_class]
    return sorted(out)


def upsample224(hm):
    """Upsample a heatmap (e.g. 7x7) to 224x224 in [0,1] via bilinear."""
    hm = np.asarray(hm, dtype=np.float64)
    if hm.shape == (224, 224):
        return np.clip(hm, 0, 1)
    im = Image.fromarray((np.clip(hm, 0, 1) * 255).astype("uint8"))
    im = im.resize((224, 224), Image.BILINEAR)
    return np.array(im) / 255.0


def normalize(x):
    """x in [0,1], shape (B,3,224,224) -> ImageNet-normalized."""
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def raw_tensor(path):
    """Load an image as a (3,224,224) tensor in [0,1] (no normalization)."""
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return t(Image.open(path).convert("RGB"))


def ssim(a, b):
    """Windowed SSIM between two [0,1] maps, scipy-only (no skimage)."""
    from scipy.ndimage import uniform_filter
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    win = 7
    mu_a = uniform_filter(a, win)
    mu_b = uniform_filter(b, win)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    va = uniform_filter(a * a, win) - mu_a2
    vb = uniform_filter(b * b, win) - mu_b2
    vab = uniform_filter(a * b, win) - mu_ab
    num = (2 * mu_ab + C1) * (2 * vab + C2)
    den = (mu_a2 + mu_b2 + C1) * (va + vb + C2)
    return float(np.mean(num / den))
