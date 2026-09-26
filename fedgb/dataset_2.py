"""
dataset_2.py
------------
Loader helpers for Dataset 2:
    yasserhessein/gallblader-diseases-dataset

This Kaggle dataset is not the same 5-class GBCU layout used by dataset.py.
It contains 9 diagnosis folders and no ready-made training/validation split,
so this module keeps its own class mapping and creates a deterministic split.
"""

import random
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset

from dataset import get_transforms


KAGGLE_DATASET = "yasserhessein/gallblader-diseases-dataset"

CLASS_FOLDERS = [
    "1Gallstones",
    "2Abdomen and retroperitoneum",
    "3cholecystitis",
    "4Membranous and gangrenous cholecystitis",
    "5Perforation",
    "6Polyps and cholesterol crystals",
    "7Adenomyomatosis",
    "8Carcinoma",
    "9Various causes of gallbladder wall thickening",
]

CLASS_NAMES = [
    "gallstones",
    "abdomen_retroperitoneum",
    "cholecystitis",
    "membranous_gangrenous_cholecystitis",
    "perforation",
    "polyps_cholesterol_crystals",
    "adenomyomatosis",
    "carcinoma",
    "wall_thickening_various",
]

CLASS_LABELS = dict(zip(CLASS_NAMES, CLASS_FOLDERS))
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
FOLDER_TO_IDX = {folder: idx for idx, folder in enumerate(CLASS_FOLDERS)}
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def download_dataset_2() -> Path:
    """
    Download Dataset 2 with KaggleHub and return the local cache path.

    Requires:
        pip install kagglehub
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Install it with: pip install kagglehub"
        ) from exc

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def find_dataset_2_root(root: str | Path) -> Path:
    """
    Find the folder that directly contains Dataset 2's 9 class folders.
    """
    root = Path(root)
    candidates = [root, *[p for p in root.rglob("*") if p.is_dir()]]

    for candidate in candidates:
        if all((candidate / folder).is_dir() for folder in CLASS_FOLDERS):
            return candidate

    raise FileNotFoundError(f"No Dataset 2 class-folder layout found under: {root}")


class GallbladderDataset2(Dataset):
    """
    Loads Dataset 2 images from its 9 class folders.
    """

    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir = find_dataset_2_root(root_dir)
        self.transform = transform
        self.samples = []

        for folder, label in FOLDER_TO_IDX.items():
            class_dir = self.root_dir / folder
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((image_path, label))

        print(f"[Dataset 2] Loaded {len(self.samples)} images from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def stratified_train_val_split(
    dataset: GallbladderDataset2,
    val_fraction: float = 0.2,
    seed: int = 42,
):
    """
    Create deterministic train/validation indices while preserving classes.
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)

    for idx, (_, label) in enumerate(dataset.samples):
        by_label[label].append(idx)

    train_indices = []
    val_indices = []
    for label, indices in by_label.items():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def dirichlet_split_2(dataset, num_clients: int, alpha: float = 0.5, seed: int = 42):
    """
    Partition Dataset 2 training data with a Dirichlet label distribution.
    """
    rng = np.random.default_rng(seed)
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]

    for label in range(NUM_CLASSES):
        indices = list(class_indices[label])
        rng.shuffle(indices)
        n = len(indices)
        if n == 0:
            continue

        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = np.floor(proportions * n).astype(int)
        remainder = n - counts.sum()
        if remainder > 0:
            for client_id in rng.choice(num_clients, size=remainder, replace=False):
                counts[client_id] += 1

        ptr = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(indices[ptr: ptr + count])
            ptr += count

    for client_id, indices in enumerate(client_indices):
        label_counts = defaultdict(int)
        for idx in indices:
            label_counts[dataset.samples[idx][1]] += 1
        detail = ", ".join(
            f"{IDX_TO_CLASS[label]}={label_counts[label]}" for label in range(NUM_CLASSES)
        )
        print(f"[Dataset 2 Dirichlet] Client {client_id}: {len(indices)} samples | {detail}")

    return client_indices


class TransformSubset(torch.utils.data.Dataset):
    """
    Subset wrapper that applies a transform different from the base dataset.
    """

    def __init__(self, dataset: GallbladderDataset2, indices, transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.samples = [dataset.samples[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_client_datasets_2(
    base_dir: str | Path | None = None,
    num_clients: int = 3,
    partition: str = "dirichlet",
    dirichlet_alpha: float = 0.5,
    val_fraction: float = 0.2,
    seed: int = 42,
):
    """
    Return Dataset 2 client train datasets plus a shared validation dataset.

    The default partition is dirichlet because Dataset 2 has 9 classes.
    """
    if base_dir is None:
        base_dir = download_dataset_2()

    raw_dataset = GallbladderDataset2(base_dir)
    train_indices, val_indices = stratified_train_val_split(
        raw_dataset, val_fraction=val_fraction, seed=seed
    )

    train_dataset = TransformSubset(
        raw_dataset, train_indices, transform=get_transforms(train=True)
    )
    val_dataset = TransformSubset(
        raw_dataset, val_indices, transform=get_transforms(train=False)
    )

    if partition == "dirichlet":
        client_indices = dirichlet_split_2(
            train_dataset, num_clients=num_clients, alpha=dirichlet_alpha, seed=seed
        )
    else:
        raise ValueError("Dataset 2 currently supports partition='dirichlet'.")

    client_datasets = [Subset(train_dataset, indices) for indices in client_indices]
    return client_datasets, val_dataset


if __name__ == "__main__":
    dataset_path = download_dataset_2()
    print("Path to dataset files:", dataset_path)
    data_root = find_dataset_2_root(dataset_path)
    print("Detected Dataset 2 root:", data_root)

    dataset = GallbladderDataset2(data_root)
    counts = defaultdict(int)
    for _, label in dataset.samples:
        counts[label] += 1

    for idx, class_name in IDX_TO_CLASS.items():
        print(f"{idx}: {class_name:38s} {counts[idx]:5d}")
