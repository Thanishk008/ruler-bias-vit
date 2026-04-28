"""Evaluate saved checkpoints and generate robustness and Grad-CAM outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.dataloader import get_dataloaders
from src.models.baseline_vit import BaselineViT
from src.models.foundation_clip import CLIPZeroShotClassifier
from src.models.swin_transformer import SwinTransformer
from src.utils import (
    compute_metrics,
    generate_gradcam,
    save_confusion_matrix,
    save_metrics_csv,
    save_pr_curve,
)


CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
ROOT = Path(__file__).resolve().parent


def _resolve_path(path: str | Path) -> Path:
    """Resolve a relative path against the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _resolve_output_dir(model_name: str, ckpt_path: Path | None) -> Path:
    """Choose a unique output directory for a given evaluation run."""
    if model_name == "foundation":
        return ROOT / "outputs" / "foundation"
    if ckpt_path is None:
        raise ValueError("A checkpoint is required for baseline and swin evaluations.")
    return ROOT / "outputs" / ckpt_path.stem


def build_model(model_name: str, num_classes: int):
    """Instantiate the requested architecture for evaluation."""
    if model_name == "baseline":
        return BaselineViT(num_classes=num_classes)
    if model_name == "swin":
        return SwinTransformer(num_classes=num_classes)
    if model_name == "foundation":
        return CLIPZeroShotClassifier(class_names=CLASS_NAMES)
    raise ValueError(f"Unsupported model: {model_name}")


def _load_checkpoint(path: Path, model, device):
    """Load model weights from a checkpoint or bare state-dict file."""
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def _evaluate_loader(model, loader, criterion, device):
    """Run inference on a dataloader and collect predictions and paths."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []
    y_prob = []
    image_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())
            image_paths.extend(paths)

    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, y_true, y_pred, y_prob, image_paths


def _maybe_build_clip_transforms(loader):
    """Apply CLIP preprocessing to an already-created dataloader."""
    transform = CLIPZeroShotClassifier.clip_transforms()
    loader.dataset.transform = transform
    return loader


def _evaluate_and_save(model, loader, criterion, device, output_dir, model_name, gradcam_samples):
    """Evaluate one loader and persist metrics, plots, and Grad-CAM images."""
    loss, y_true, y_pred, y_prob, image_paths = _evaluate_loader(model, loader, criterion, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, CLASS_NAMES)

    save_confusion_matrix(y_true, y_pred, CLASS_NAMES, output_dir / "confusion_matrix.png")
    save_pr_curve(y_true, y_prob, CLASS_NAMES, output_dir / "pr_curve.png")
    save_metrics_csv(metrics, output_dir / "metrics.csv")

    gradcam_dir = output_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    misclassified = [
        (idx, true, pred, path)
        for idx, (true, pred, path) in enumerate(zip(y_true, y_pred, image_paths))
        if true != pred
    ]
    for sample_idx, true_label, pred_label, path in misclassified[:gradcam_samples]:
        matches = loader.dataset.df.index[loader.dataset.df["image_path"] == path].tolist()
        if not matches:
            continue
        image, _, _ = loader.dataset[matches[0]]
        image_tensor = image.unsqueeze(0).to(device)
        save_path = gradcam_dir / f"sample_{sample_idx}_true_{true_label}_pred_{pred_label}.png"
        generate_gradcam(model, image_tensor, pred_label, save_path, model_name)

    return loss, metrics


def main():
    """Run evaluation on the test split and optional robustness splits."""
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier.")
    parser.add_argument("--model", choices=["baseline", "swin", "foundation"], default="swin")
    parser.add_argument("--data_root", default="data/isic2019")
    parser.add_argument("--splits_dir", default="data/isic2019/splits/")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path for baseline or swin. Ignored for foundation.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robustness_test", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradcam_samples", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data_root = _resolve_path(args.data_root)
    splits_dir = _resolve_path(args.splits_dir)
    default_ckpts = {
        "baseline": "models/baseline_none_best.pth",
        "swin": "models/swin_none_best.pth",
    }
    ckpt_path = _resolve_path(args.ckpt) if args.ckpt is not None else (
        _resolve_path(default_ckpts[args.model]) if args.model in default_ckpts else None
    )

    dataloaders, _ = get_dataloaders(
        data_root=data_root,
        splits_dir=splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.model == "foundation":
        for split in dataloaders:
            dataloaders[split] = _maybe_build_clip_transforms(dataloaders[split])

    model = build_model(args.model, num_classes=len(CLASS_NAMES)).to(device)
    if args.model == "foundation":
        if args.ckpt is not None:
            print("Ignoring --ckpt for foundation; zero-shot CLIP inference does not use a checkpoint.")
    else:
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for {args.model}. Provide a trained checkpoint with --ckpt, or place it at {default_ckpts.get(args.model, 'models/<model>_none_best.pth')}."
            )
        _load_checkpoint(ckpt_path, model, device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    output_dir = _resolve_output_dir(args.model, ckpt_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    _test_loss, test_metrics = _evaluate_and_save(
        model,
        dataloaders["test"],
        criterion,
        device,
        output_dir,
        args.model,
        args.gradcam_samples,
    )

    metrics_table = pd.DataFrame(
        [
            {
                "class": class_name,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
            }
            for class_name, scores in test_metrics["per_class"].items()
        ]
    )
    metrics_table.loc[len(metrics_table)] = {
        "class": "macro",
        "precision": test_metrics["macro_precision"],
        "recall": test_metrics["macro_recall"],
        "f1": test_metrics["macro_f1"],
    }
    print(metrics_table.to_string(index=False))

    if args.robustness_test:
        comparison = {
            "Metric": ["Macro Recall", "Macro F1", "Accuracy"],
            "Full Test": [
                test_metrics["macro_recall"],
                test_metrics["macro_f1"],
                test_metrics["accuracy"],
            ],
            "No Ruler": [np.nan, np.nan, np.nan],
            "With Ruler": [np.nan, np.nan, np.nan],
        }
        for split_name, label in [("test_no_ruler", "No Ruler"), ("test_with_ruler", "With Ruler")]:
            _, metrics = _evaluate_and_save(
                model,
                dataloaders[split_name],
                criterion,
                device,
                output_dir / split_name,
                args.model,
                0,
            )
            comparison[label] = [metrics["macro_recall"], metrics["macro_f1"], metrics["accuracy"]]

        comparison_df = pd.DataFrame(comparison, columns=["Metric", "Full Test", "No Ruler", "With Ruler"])
        comparison_df.to_csv(output_dir / "robustness_comparison.csv", index=False)
        print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
