"""Dataset and dataloader helpers for the ISIC 2019 skin lesion task."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 8


class ISICDataset(Dataset):
    """PyTorch dataset for ISIC image/label CSV splits."""

    def __init__(self, csv_path, data_root, transform=None):
        """Load a split CSV and prepare image access."""
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform if transform is not None else get_transforms("val")
        self.df = pd.read_csv(self.csv_path)
        required_columns = {"image_path", "label"}
        missing = required_columns.difference(self.df.columns)
        if missing:
            raise ValueError(f"CSV {self.csv_path} is missing required columns: {sorted(missing)}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index):
        """Return an image tensor, label, and image path string."""
        row = self.df.iloc[index]
        image_path = self.data_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), str(row["image_path"])


def get_transforms(split: str) -> T.Compose:
    """Build the torchvision transform pipeline for a dataset split."""
    split = split.lower()
    if split == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(20),
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    if split in {"val", "test", "test_no_ruler", "test_with_ruler"}:
        return T.Compose(
            [
                T.Resize(224, interpolation=InterpolationMode.BILINEAR),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    raise ValueError(f"Unsupported split: {split}")


def _load_split_csv(path: Path, fallback: Path | None = None) -> pd.DataFrame:
    """Load a CSV split, optionally falling back to another split."""
    if path.exists():
        return pd.read_csv(path)
    if fallback is not None and fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"Could not find split CSV at {path}")


def get_dataloaders(data_root, splits_dir, batch_size, num_workers=4) -> Tuple[Dict[str, DataLoader], torch.Tensor]:
    """Create dataloaders for every saved split and compute class weights."""
    data_root = Path(data_root)
    splits_dir = Path(splits_dir)

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    test_no_ruler_csv = splits_dir / "test_no_ruler.csv"
    test_with_ruler_csv = splits_dir / "test_with_ruler.csv"

    train_df = _load_split_csv(train_csv)
    val_df = _load_split_csv(val_csv, fallback=test_csv)
    test_df = _load_split_csv(test_csv)
    test_no_ruler_df = _load_split_csv(test_no_ruler_csv, fallback=test_csv)
    test_with_ruler_df = _load_split_csv(test_with_ruler_csv, fallback=test_csv)

    datasets = {
        "train": ISICDataset(train_csv, data_root, transform=get_transforms("train")),
        "val": ISICDataset(val_csv if val_csv.exists() else test_csv, data_root, transform=get_transforms("val")),
        "test": ISICDataset(test_csv, data_root, transform=get_transforms("test")),
        "test_no_ruler": ISICDataset(
            test_no_ruler_csv if test_no_ruler_csv.exists() else test_csv,
            data_root,
            transform=get_transforms("test_no_ruler"),
        ),
        "test_with_ruler": ISICDataset(
            test_with_ruler_csv if test_with_ruler_csv.exists() else test_csv,
            data_root,
            transform=get_transforms("test_with_ruler"),
        ),
    }

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(num_workers > 0),
    }
    if num_workers == 0:
        loader_kwargs["persistent_workers"] = False

    dataloaders = {
        split: DataLoader(dataset, shuffle=(split == "train"), **loader_kwargs)
        for split, dataset in datasets.items()
    }

    class_counts = np.bincount(train_df["label"].astype(int).to_numpy(), minlength=NUM_CLASSES).astype(np.float32)
    total_samples = float(class_counts.sum())
    class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    valid = class_counts > 0
    class_weights[valid] = total_samples / (NUM_CLASSES * class_counts[valid])

    return dataloaders, torch.tensor(class_weights, dtype=torch.float32)
