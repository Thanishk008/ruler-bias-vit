"""Create ISIC 2019 train/val/test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

LABEL_COLUMNS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
UNKNOWN_COLUMN = "UNK"


def _resolve_path(path: str, base: Path) -> Path:
    """Resolve a possibly relative path against the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _build_split_frame(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Attach a split column to a frame while preserving image paths and labels."""
    result = df.copy().reset_index(drop=True)
    result["split"] = split_name
    return result[["image_path", "label", "split"]]


def _dataset_is_present(data_root: Path) -> bool:
    """Check whether the expected Kaggle dataset layout is already available."""
    return (data_root / "ISIC_2019_Training_GroundTruth.csv").exists() and (
        data_root / "ISIC_2019_Training_Input"
    ).exists()


def _download_kaggle_dataset(dataset: str, data_root: Path, force: bool = False) -> None:
    """Download the Kaggle dataset directly into the expected data root."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "The kagglehub package is required for dataset downloads. Install it with `pip install kagglehub`."
        ) from exc

    data_root.mkdir(parents=True, exist_ok=True)
    kagglehub.dataset_download(dataset, output_dir=str(data_root), force_download=force)


def main():
    """Parse arguments, split metadata, and save the resulting CSV files."""
    parser = argparse.ArgumentParser(description="Prepare ISIC 2019 CSV splits.")
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Path to ISIC_2019_Training_GroundTruth.csv. "
            "Defaults to <data_root>/ISIC_2019_Training_GroundTruth.csv if omitted."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/isic2019/splits/",
        help="Directory to save split CSVs. Defaults to data/isic2019/splits/.",
    )
    parser.add_argument(
        "--data_root",
        default="data/isic2019",
        help="Root directory containing the downloaded ISIC 2019 files. Defaults to data/isic2019.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download_kaggle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the Kaggle dataset into data_root if it is missing.",
    )
    parser.add_argument(
        "--kaggle_dataset",
        default="andrewmvd/isic-2019",
        help="Kaggle dataset slug to download when --download_kaggle is set.",
    )
    parser.add_argument(
        "--kaggle_force",
        action="store_true",
        help="Force Kaggle to re-download files even if they already exist.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_root = _resolve_path(args.data_root, project_root)
    csv_path = _resolve_path(args.csv, project_root) if args.csv is not None else data_root / "ISIC_2019_Training_GroundTruth.csv"
    output_dir = _resolve_path(args.output, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing ISIC 2019 splits in {output_dir}")
    if args.download_kaggle and (args.kaggle_force or not _dataset_is_present(data_root)):
        print(f"Downloading Kaggle dataset `{args.kaggle_dataset}` into {data_root}")
        _download_kaggle_dataset(args.kaggle_dataset, data_root, force=args.kaggle_force)
    else:
        print(f"Using existing dataset at {data_root}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find metadata CSV at {csv_path}. "
            "Download ISIC 2019 into data_root first, or let this script download it with kagglehub."
        )

    image_root = data_root / "ISIC_2019_Training_Input"
    if not image_root.exists():
        raise FileNotFoundError(
            f"Could not find image directory at {image_root}. "
            "Make sure the Kaggle files were downloaded into the expected folder."
        )

    metadata = pd.read_csv(csv_path)
    required_columns = {"image"} | set(LABEL_COLUMNS) | {UNKNOWN_COLUMN}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata CSV must contain columns {sorted(required_columns)}; missing {sorted(missing)}")

    metadata = metadata.copy()
    unknown_mask = metadata[UNKNOWN_COLUMN].fillna(0).astype(int) > 0
    metadata = metadata.loc[~unknown_mask].reset_index(drop=True)
    if metadata.empty:
        raise ValueError("No labeled samples found after removing UNK rows.")

    label_matrix = metadata[LABEL_COLUMNS].fillna(0).astype(int)
    label_sums = label_matrix.sum(axis=1)
    if (label_sums == 0).any():
        raise ValueError("Found samples with no positive label among the 8 ISIC 2019 classes.")

    metadata["label"] = label_matrix.values.argmax(axis=1)
    metadata["image_path"] = metadata["image"].astype(str).map(lambda x: Path("ISIC_2019_Training_Input") / f"{x}.jpg")
    metadata["image_path"] = metadata["image_path"].map(lambda p: p.as_posix())

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, temp_idx = next(splitter.split(metadata, metadata["label"]))
    train_df = metadata.iloc[train_idx].reset_index(drop=True)
    temp_df = metadata.iloc[temp_idx].reset_index(drop=True)

    temp_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    val_idx, test_idx = next(temp_splitter.split(temp_df, temp_df["label"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    train_out = _build_split_frame(train_df, "train")
    val_out = _build_split_frame(val_df, "val")
    test_out = _build_split_frame(test_df, "test")

    train_out.to_csv(output_dir / "train.csv", index=False)
    val_out.to_csv(output_dir / "val.csv", index=False)
    test_out.to_csv(output_dir / "test.csv", index=False)

    _build_split_frame(test_df, "test_no_ruler").to_csv(output_dir / "test_no_ruler.csv", index=False)
    _build_split_frame(test_df, "test_with_ruler").to_csv(output_dir / "test_with_ruler.csv", index=False)

    print(
        "Saved splits: "
        f"train={len(train_out)}, val={len(val_out)}, test={len(test_out)}, "
        f"test_no_ruler={len(test_out)}, test_with_ruler={len(test_out)}"
    )


if __name__ == "__main__":
    main()
