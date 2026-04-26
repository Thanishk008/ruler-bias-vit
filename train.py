"""Train the skin-lesion classifiers with optional debiasing techniques."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torchvision import transforms as T

from src.dataloader import ISICDataset, get_dataloaders, get_transforms
from src.models.baseline_vit import BaselineViT
from src.models.foundation_clip import CLIPLinearProbe
from src.models.swin_transformer import SwinTransformer
from src.techniques.technique1_debiasing import RulerCropTransform, SyntheticRulerAugmentation
from src.techniques.technique2_attention_reg import AttentionRegularizationLoss
from src.techniques.technique3_patch_masking import wrap_model_with_masker
from src.utils import compute_metrics


CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
ROOT = Path(__file__).resolve().parent


def _resolve_path(path: str | Path) -> Path:
    """Resolve a relative path against the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _str2bool(value):
    """Parse flexible boolean command-line values."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def set_seed(seed: int):
    """Seed every random number generator used in training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, num_classes: int, pretrained: bool):
    """Instantiate the requested backbone."""
    if model_name == "baseline":
        return BaselineViT(num_classes=num_classes, pretrained=pretrained)
    if model_name == "swin":
        return SwinTransformer(num_classes=num_classes, pretrained=pretrained)
    if model_name == "foundation":
        return CLIPLinearProbe(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def _compose_with_technique1(transform):
    """Inject ruler-specific preprocessing into a torchvision transform."""
    transforms_list = list(transform.transforms if isinstance(transform, T.Compose) else [transform])
    injected = []
    inserted = False
    for op in transforms_list:
        if not inserted and isinstance(op, T.ToTensor):
            injected.extend([RulerCropTransform(), SyntheticRulerAugmentation()])
            inserted = True
        injected.append(op)
    if not inserted:
        injected.extend([RulerCropTransform(), SyntheticRulerAugmentation()])
    return T.Compose(injected)


def _apply_transforms_to_loader(loader, transform):
    """Replace the underlying dataset transform in a dataloader."""
    loader.dataset.transform = transform
    return loader


def _build_clip_transforms(train: bool):
    """Return CLIP preprocessing, optionally with simple augmentation."""
    base = CLIPLinearProbe.clip_transforms()
    if not train:
        return base
    transforms_list = list(base.transforms)
    augmented = []
    inserted = False
    for op in transforms_list:
        if not inserted and isinstance(op, T.ToTensor):
            augmented.extend(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomRotation(20),
                    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                ]
            )
            inserted = True
        augmented.append(op)
    return T.Compose(augmented)


def _evaluate(model, loader, criterion, device):
    """Run a validation loop and return averaged loss and predictions."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []
    y_prob = []

    use_amp = device.type == "cuda"
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
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

    metrics = compute_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
    return total_loss / max(1, total_samples), metrics, y_true, y_pred, y_prob


def _save_checkpoint(path: Path, epoch: int, model, optimizer, scheduler, best_val_recall: float, args):
    """Persist the training state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_recall": best_val_recall,
            "args": vars(args),
        },
        path,
    )


def _load_checkpoint(path: Path, model, optimizer, scheduler, device):
    """Restore model, optimizer, and scheduler state from a checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_val_recall", 0.0))
    model.load_state_dict(checkpoint)
    return 0, 0.0


def main():
    """Train the requested model and save the best checkpoint."""
    parser = argparse.ArgumentParser(description="Train debiased lesion classifiers.")
    parser.add_argument("--model", choices=["baseline", "swin", "foundation"], default="swin")
    parser.add_argument("--technique", choices=["none", "technique1", "technique2", "technique3"], default="none")
    parser.add_argument("--data_root", default="data/isic2019")
    parser.add_argument("--splits_dir", default="data/isic2019/splits/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="outputs/")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", type=_str2bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable mixed precision on CUDA.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data_root = _resolve_path(args.data_root)
    splits_dir = _resolve_path(args.splits_dir)
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)

    dataloaders, class_weights = get_dataloaders(
        data_root=data_root,
        splits_dir=splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, num_classes=len(CLASS_NAMES), pretrained=args.pretrained)
    model = model.to(device)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.model == "foundation":
        args.lr = 1e-3
        trainable_params = model.classifier.parameters()
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    masker = None
    if args.technique == "technique3":
        if args.model == "foundation":
            print("technique3 is not compatible with the CLIP linear probe; skipping patch masking.")
        else:
            model, masker = wrap_model_with_masker(model, mask_prob=0.5)

    if args.model == "foundation":
        clip_train = _build_clip_transforms(train=True)
        clip_eval = _build_clip_transforms(train=False)
        dataloaders["train"] = _apply_transforms_to_loader(dataloaders["train"], clip_train)
        for split in ["val", "test", "test_no_ruler", "test_with_ruler"]:
            dataloaders[split] = _apply_transforms_to_loader(dataloaders[split], clip_eval)

    if args.technique == "technique1":
        base_train_transform = dataloaders["train"].dataset.transform
        train_transform = _compose_with_technique1(base_train_transform)
        dataloaders["train"] = _apply_transforms_to_loader(dataloaders["train"], train_transform)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    attention_regularizer = AttentionRegularizationLoss(lambda_weight=0.1)
    use_attention_regularization = args.technique == "technique2" and args.model in {"baseline", "swin"}
    if args.technique == "technique2" and not use_attention_regularization:
        print("technique2 is not compatible with the selected model; skipping attention regularization loss.")

    start_epoch = 1
    best_val_recall = float("-inf")
    if args.resume:
        resume_path = _resolve_path(args.resume)
        last_epoch, best_val_recall = _load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch = last_epoch + 1

    best_model_path = ROOT / "models" / f"{args.model}_{args.technique}_best.pth"
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0

        train_loader = tqdm(dataloaders["train"], desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                ce_loss = criterion(logits, labels)
                loss = ce_loss
                if use_attention_regularization:
                    attention_weights = model.get_attention_weights(images)
                    loss = ce_loss + attention_regularizer(attention_weights)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size
            train_loader.set_postfix(loss=running_loss / max(1, running_samples))

        train_loss = running_loss / max(1, running_samples)
        val_loss, val_metrics, _, _, _ = _evaluate(model, dataloaders["val"], criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Recall (macro): {val_metrics['macro_recall']:.4f} | Val F1 (macro): {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_recall"] > best_val_recall:
            best_val_recall = val_metrics["macro_recall"]
            _save_checkpoint(best_model_path, epoch, model, optimizer, scheduler, best_val_recall, args)

        if epoch % args.save_every == 0:
            _save_checkpoint(checkpoint_dir / f"{args.model}_{args.technique}_epoch_{epoch}.pth", epoch, model, optimizer, scheduler, best_val_recall, args)

    if masker is not None:
        masker.remove_hook()


if __name__ == "__main__":
    main()
