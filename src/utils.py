"""Shared evaluation, visualisation, and Grad-CAM utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .techniques.technique2_attention_reg import BORDER_PATCH_INDICES


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def reshape_transform_vit(tensor, height=14, width=14):
    """Reshape ViT tokens into a spatial feature map for Grad-CAM."""
    batch_size, _, channels = tensor.shape
    result = tensor[:, 1:, :].reshape(batch_size, height, width, channels)
    return result.transpose(2, 3).transpose(1, 2)


def reshape_transform_swin(tensor, height=7, width=7):
    """Reshape Swin tokens into a spatial feature map for Grad-CAM."""
    batch_size, _, channels = tensor.shape
    result = tensor.reshape(batch_size, height, width, channels)
    return result.transpose(2, 3).transpose(1, 2)


def _denormalize_image(image_tensor: torch.Tensor, model_type: str) -> np.ndarray:
    """Convert a normalized tensor back into an RGB image in [0, 1]."""
    image = image_tensor.detach().float().cpu().clone()
    if image.ndim == 4:
        image = image[0]
    if model_type == "foundation":
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    image = image.permute(1, 2, 0).numpy()
    image = image * std + mean
    return np.clip(image, 0.0, 1.0)


def generate_gradcam(model, image_tensor, target_class, save_path, model_type):
    """Generate and save a Grad-CAM overlay or a fallback placeholder."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "foundation":
        canvas = Image.new("RGB", (224, 224), color="white")
        draw = ImageDraw.Draw(canvas)
        message = "Grad-CAM not supported for CLIP"
        bbox = draw.textbbox((0, 0), message)
        x = (224 - (bbox[2] - bbox[0])) // 2
        y = (224 - (bbox[3] - bbox[1])) // 2
        draw.text((x, y), message, fill="black")
        canvas.save(save_path)
        return

    if model_type == "baseline":
        target_layers = [model.model.blocks[-1].norm1]
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_vit)
    elif model_type == "swin":
        target_layers = [model.model.layers[-1].blocks[-1].norm1]
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_swin)
    else:
        raise ValueError(f"Unsupported model_type for Grad-CAM: {model_type}")

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
    rgb_image = _denormalize_image(image_tensor, model_type)
    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    Image.fromarray(cam_image).save(save_path)


def compute_metrics(y_true, y_pred, y_prob, class_names) -> Dict[str, object]:
    """Compute per-class and macro-averaged classification metrics."""
    labels = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    accuracy = float(accuracy_score(y_true, y_pred))

    per_class = {
        class_name: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
        }
        for idx, class_name in enumerate(class_names)
    }
    return {
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Save a normalized confusion matrix plot."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, f"{cm[row, col]:.2f}", ha="center", va="center", color="black", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_pr_curve(y_true, y_prob, class_names, save_path):
    """Save a one-vs-rest precision-recall curve for every class."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, class_name in enumerate(class_names):
        binary_true = (y_true == idx).astype(int)
        if binary_true.sum() == 0:
            ap = 0.0
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label=f"{class_name} (AP={ap:.3f})")
            continue
        precision, recall, _ = precision_recall_curve(binary_true, y_prob[:, idx])
        ap = average_precision_score(binary_true, y_prob[:, idx])
        ax.plot(recall, precision, linewidth=2, label=f"{class_name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("One-vs-Rest Precision-Recall Curves")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_metrics_csv(metrics_dict, save_path):
    """Save per-class metrics and a macro-average row to CSV."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for class_name, scores in metrics_dict["per_class"].items():
        rows.append(
            {
                "class": class_name,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
            }
        )
    rows.append(
        {
            "class": "macro",
            "precision": metrics_dict["macro_precision"],
            "recall": metrics_dict["macro_recall"],
            "f1": metrics_dict["macro_f1"],
        }
    )
    pd.DataFrame(rows).to_csv(save_path, index=False)
