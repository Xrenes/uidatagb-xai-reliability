"""
dataset.py
----------
Loads the UIdataGB ("Gallblader Diseases Dataset") -- 9 classes, split by
split_uidatagb.py (70/30 stratified, seed=42; no patient IDs available for
this dataset either, so this is an image-level split, same caveat as GBCU).

Dataset structure (after split_uidatagb.py):
    fedgb/data/uidatagb/
        training/
            01_gallstones/
            02_abdomen_and_retroperitoneum/
            03_cholecystitis/
            04_membranous_and_gangrenous_cholecystitis/
            05_perforation/
            06_polyps_and_cholesterol_crystals/
            07_adenomyomatosis/
            08_carcinoma/
            09_various_causes_of_gallbladder_wall_thickening/
        validation/
            <same 9 subfolders>

Kaggle: yasserhessein/gallblader-diseases-dataset
Source: UIdataGB (Turki et al., Mendeley Data, DOI 10.17632/r6h24d2d3y.1)

NOTE: this file previously loaded the GBCU dataset (5 classes: nml, bmt,
stn, abn, malg). That study is archived in full, working order under
GBCU_Study/ at the project root (paper + code + checkpoints + data) --
see GBCU_Study/fedgb/dataset.py for the original 5-class version.
"""

import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image


# ── Label mapping ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
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
CLASS_LABELS = {
    "01_gallstones": "Gallstones",
    "02_abdomen_and_retroperitoneum": "Abdomen & Retroperitoneum",
    "03_cholecystitis": "Cholecystitis",
    "04_membranous_and_gangrenous_cholecystitis": "Membranous/Gangrenous Cholecystitis",
    "05_perforation": "Perforation",
    "06_polyps_and_cholesterol_crystals": "Polyps & Cholesterol Crystals",
    "07_adenomyomatosis": "Adenomyomatosis",
    "08_carcinoma": "Carcinoma",
    "09_various_causes_of_gallbladder_wall_thickening": "Wall Thickening (Various Causes)",
}
CLASS_SHORT = {
    "01_gallstones": "GS",
    "02_abdomen_and_retroperitoneum": "ART",
    "03_cholecystitis": "CHOL",
    "04_membranous_and_gangrenous_cholecystitis": "MGC",
    "05_perforation": "PERF",
    "06_polyps_and_cholesterol_crystals": "PCC",
    "07_adenomyomatosis": "ADNM",
    "08_carcinoma": "CARC",
    "09_various_causes_of_gallbladder_wall_thickening": "WT",
}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES  = len(CLASS_NAMES)  # 9


# ── Transforms ───────────────────────────────────────────────────────────────
def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset class ─────────────────────────────────────────────────────────────
class GBCUDataset(Dataset):
    """
    Loads images from a split folder (training/ or validation/).
    Expects subfolders named: nml, bmt, stn, abn, malg
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []   # list of (image_path, label_idx)

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[WARNING] Folder not found: {class_dir} — skipping.")
                continue
            label = CLASS_TO_IDX[class_name]
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(class_dir, fname), label))

        print(f"[Dataset] Loaded {len(self.samples)} images from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Non-IID Federated Split ───────────────────────────────────────────────────
def non_iid_split(dataset: GBCUDataset, num_clients: int = 3, seed: int = 42):
    """
    Splits the training dataset into non-IID partitions per client.

    Simulated hospital biases (5 classes: nml, bmt, stn, abn, malg):
        Client 0 (Hospital A — community):  heavy normal, light malignant
        Client 1 (Hospital B — GI clinic):  heavy stones + benign, light malignant
        Client 2 (Hospital C — oncology):   heavy malignant + abnormal, light normal

    Returns a list of index lists — one per client.
    """
    random.seed(seed)

    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    for label in class_indices:
        random.shuffle(class_indices[label])

    # Relative weights per client per class: [nml, bmt, stn, abn, malg]
    # These are normalized per class so ALL images are distributed.
    # Higher weight = this hospital sees more of that class.
    client_weights = [
        [0.60, 0.10, 0.15, 0.10, 0.05],   # Hospital A — mostly normal
        [0.10, 0.30, 0.40, 0.15, 0.05],   # Hospital B — mostly benign (stones+bmt)
        [0.05, 0.10, 0.10, 0.25, 0.60],   # Hospital C — mostly malignant
    ]

    assert num_clients == len(client_weights), \
        "Update client_weights if changing num_clients"

    client_indices = [[] for _ in range(num_clients)]

    for label in range(NUM_CLASSES):
        indices = class_indices[label]
        total = len(indices)
        # Normalise weights for this class so they sum to 1 → all images used
        col = [client_weights[c][label] for c in range(num_clients)]
        col_sum = sum(col)
        normed = [w / col_sum for w in col]
        ptr = 0
        for c in range(num_clients - 1):
            count = int(total * normed[c])
            client_indices[c].extend(indices[ptr:ptr + count])
            ptr += count
        # Last client gets all remaining images (avoids rounding loss)
        client_indices[num_clients - 1].extend(indices[ptr:])

    for c, idxs in enumerate(client_indices):
        label_counts = defaultdict(int)
        for i in idxs:
            label_counts[dataset.samples[i][1]] += 1
        detail = ", ".join(f"{CLASS_NAMES[k]}={label_counts[k]}" for k in range(NUM_CLASSES))
        print(f"[Split] Client {c}: {len(idxs)} samples | {detail}")

    return client_indices


# ── Dirichlet Non-IID split (supports any N clients) ─────────────────────────
def dirichlet_split(dataset: GBCUDataset, num_clients: int,
                    alpha: float = 0.5, seed: int = 42):
    """
    Partition training data using a Dirichlet(alpha) label distribution.
    Standard NIID-Bench approach — smaller alpha → more heterogeneous.

    Args:
        dataset:     GBCUDataset (training).
        num_clients: Number of simulated clients (any value ≥ 1).
        alpha:       Dirichlet concentration parameter (0.1 = very non-IID,
                     100 = near-IID).
        seed:        Random seed.

    Returns:
        List of index lists — one per client.
    """
    rng = np.random.default_rng(seed)

    class_indices: defaultdict = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]

    for label in range(NUM_CLASSES):
        indices = list(class_indices[label])
        rng.shuffle(indices)
        n = len(indices)
        if n == 0:
            continue
        # Draw proportions from Dir(alpha * 1_K)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = np.floor(proportions * n).astype(int)
        # Distribute rounding remainder
        remainder = n - counts.sum()
        for k in rng.choice(num_clients, size=remainder, replace=False):
            counts[k] += 1
        ptr = 0
        for c in range(num_clients):
            client_indices[c].extend(indices[ptr: ptr + counts[c]])
            ptr += counts[c]

    for c, idxs in enumerate(client_indices):
        lc: defaultdict = defaultdict(int)
        for i in idxs:
            lc[dataset.samples[i][1]] += 1
        detail = ", ".join(f"{CLASS_NAMES[k]}={lc[k]}" for k in range(NUM_CLASSES))
        print(f"[Dirichlet] Client {c:3d}: {len(idxs):4d} samples | {detail}")

    return client_indices


# ── Mixed-skew split (label skew + quantity skew) ─────────────────────────────
def mixed_skew_split(dataset: GBCUDataset, num_clients: int,
                     alpha_label: float = 0.5, alpha_qty: float = 0.3,
                     noise_std: float = 0.0, seed: int = 42):
    """
    Combined label skew + quantity skew partition.

    Steps:
      1. Assign class distributions via Dirichlet(alpha_label).
      2. Apply quantity skew: each client retains only a Dirichlet(alpha_qty)
         fraction of its assigned samples (total samples preserved globally
         but per-client sizes vary dramatically).
      3. Optionally record per-client noise level for feature-skew simulation
         (actual image noise is applied in the DataLoader wrapper).

    Returns:
        (client_indices, client_noise_levels)
        client_noise_levels: list of per-client additive Gaussian noise std
                             (0 if noise_std == 0).
    """
    rng = np.random.default_rng(seed)

    # Step 1 — label skew
    label_indices = dirichlet_split(dataset, num_clients,
                                    alpha=alpha_label, seed=seed)

    # Step 2 — quantity skew: redistribute total via Dir(alpha_qty)
    total = sum(len(idx) for idx in label_indices)
    qty_props = rng.dirichlet(np.full(num_clients, alpha_qty))
    target_sizes = np.floor(qty_props * total).astype(int)
    # Ensure every client has at least 1 sample
    target_sizes = np.maximum(target_sizes, 1)
    # Fix rounding
    diff = total - target_sizes.sum()
    for k in rng.choice(num_clients, size=abs(diff), replace=False):
        target_sizes[k] += int(np.sign(diff))

    mixed_indices = []
    for c, indices in enumerate(label_indices):
        n = min(len(indices), int(target_sizes[c]))
        arr = np.array(indices)
        rng.shuffle(arr)
        mixed_indices.append(arr[:n].tolist())

    # Step 3 — per-client feature-skew noise levels
    if noise_std > 0:
        # Draw noise levels ~ Uniform(0, noise_std) per client
        client_noise = rng.uniform(0, noise_std, num_clients).tolist()
    else:
        client_noise = [0.0] * num_clients

    for c, idxs in enumerate(mixed_indices):
        lc: defaultdict = defaultdict(int)
        for i in idxs:
            lc[dataset.samples[i][1]] += 1
        detail = ", ".join(f"{CLASS_NAMES[k]}={lc[k]}" for k in range(NUM_CLASSES))
        print(f"[MixedSkew] Client {c:3d}: {len(idxs):4d} samples | "
              f"noise={client_noise[c]:.3f} | {detail}")

    return mixed_indices, client_noise


# ── Noisy dataset wrapper (feature-skew simulation) ───────────────────────────
class NoisySubset(torch.utils.data.Dataset):
    """Wraps a Subset and adds additive Gaussian noise to every image tensor."""

    def __init__(self, subset: Subset, noise_std: float = 0.0):
        self.subset = subset
        self.noise_std = noise_std

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.noise_std > 0:
            img = img + torch.randn_like(img) * self.noise_std
        return img, label


def get_client_datasets(base_dir: str, num_clients: int = 3,
                        partition: str = "label_skew",
                        dirichlet_alpha: float = 0.5,
                        mixed_alpha_label: float = 0.5,
                        mixed_alpha_qty: float = 0.3,
                        mixed_noise_std: float = 0.05,
                        seed: int = 42):
    """
    Returns train datasets per client + a shared validation dataset.

    Args:
        base_dir:    Folder containing training/ and validation/.
        num_clients: Number of simulated clients.
        partition:   One of:
                       "label_skew"   — 3-client hardcoded hospital weights (default)
                       "dirichlet"    — Dirichlet(alpha) for any N clients
                       "mixed_skew"   — Dirichlet label + quantity skew + feature noise
        dirichlet_alpha:   Concentration for "dirichlet" partition.
        mixed_alpha_label: Label-skew alpha for "mixed_skew".
        mixed_alpha_qty:   Quantity-skew alpha for "mixed_skew".
        mixed_noise_std:   Per-client Gaussian image noise std for "mixed_skew".
        seed:        Random seed.

    Returns:
        (client_datasets, test_dataset)
    """
    train_dir = os.path.join(base_dir, "training")
    val_dir   = os.path.join(base_dir, "validation")

    train_dataset = GBCUDataset(train_dir, transform=get_transforms(train=True))
    test_dataset  = GBCUDataset(val_dir,   transform=get_transforms(train=False))

    client_noise_levels = [0.0] * num_clients  # default: no feature noise

    if partition == "label_skew":
        if num_clients != 3:
            print(f"[WARNING] label_skew is hardcoded for 3 clients. "
                  f"Falling back to dirichlet for num_clients={num_clients}.")
            client_indices = dirichlet_split(train_dataset, num_clients,
                                             alpha=dirichlet_alpha, seed=seed)
        else:
            client_indices = non_iid_split(train_dataset, num_clients=3)
    elif partition == "dirichlet":
        client_indices = dirichlet_split(train_dataset, num_clients,
                                         alpha=dirichlet_alpha, seed=seed)
    elif partition == "mixed_skew":
        client_indices, client_noise_levels = mixed_skew_split(
            train_dataset, num_clients,
            alpha_label=mixed_alpha_label,
            alpha_qty=mixed_alpha_qty,
            noise_std=mixed_noise_std,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown partition '{partition}'. "
                         "Choose from: label_skew, dirichlet, mixed_skew")

    client_datasets = []
    for c, idxs in enumerate(client_indices):
        subset = Subset(train_dataset, idxs)
        if client_noise_levels[c] > 0:
            client_datasets.append(NoisySubset(subset, client_noise_levels[c]))
        else:
            client_datasets.append(subset)

    return client_datasets, test_dataset
