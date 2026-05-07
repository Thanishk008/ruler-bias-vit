from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.models.baseline_vit import BaselineViT
from src.models.swin_transformer import SwinTransformer


OUT = ROOT / "outputs" / "report_assets" / "model_architecture_summary_v4.png"


def count_params(model) -> str:
    total = sum(p.numel() for p in model.parameters())
    return f"{total / 1_000_000:.2f}M"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    baseline = BaselineViT(num_classes=8).model
    swin = SwinTransformer(num_classes=8).model

    fig = plt.figure(figsize=(16, 8.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Model-Summary Style Architecture Snapshot",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.925,
        "Generated from the actual PyTorch/timm model definitions used in this project",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )

    def add_box(x: float, y: float, w: float, h: float, title: str, lines: list[str], fc: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#1f2937",
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.035, title, ha="left", va="top", fontsize=13, fontweight="bold")
        text_y = y + h - 0.085
        for line in lines:
            ax.text(x + 0.02, text_y, line, ha="left", va="top", fontsize=10.5)
            text_y -= 0.052

    add_box(
        0.03,
        0.13,
        0.38,
        0.71,
        "Baseline ViT-B/16",
        [
            "Input: RGB image, 224 x 224",
            "Patch embedding: Conv2d kernel/stride 16 -> 196 patches",
            "Token sequence: 196 image tokens + 1 CLS token",
            "Encoder depth: 12 transformer blocks",
            "Attention: 12 heads, embed dim 768",
            "MLP hidden dim: 3072",
            f"Parameters: {count_params(BaselineViT(num_classes=8))} total / {count_params(BaselineViT(num_classes=8))} trainable",
            "Classifier: linear head -> 8 logits",
            "Purpose: shared scratch-trained baseline",
        ],
        "#e7f0fb",
    )

    add_box(
        0.59,
        0.13,
        0.38,
        0.71,
        "Proposed Swin-Tiny",
        [
            "Input: RGB image, 224 x 224",
            "Patch embedding: Conv2d kernel/stride 4",
            "Hierarchy: 4 stages with patch merging between stages",
            "Stage 1: depth 2, dim 96, 3 heads",
            "Stage 2: depth 2, dim 192, 6 heads",
            "Stage 3: depth 6, dim 384, 12 heads",
            "Stage 4: depth 2, dim 768, 24 heads",
            "Window attention: 7 x 7 shifted windows",
            f"Parameters: {count_params(SwinTransformer(num_classes=8))} total / {count_params(SwinTransformer(num_classes=8))} trainable",
            "Classifier: global average pooling + 8-class head",
        ],
        "#e8f6ee",
    )

    ax.annotate(
        "",
        xy=(0.59, 0.50),
        xytext=(0.41, 0.50),
        arrowprops=dict(arrowstyle="<->", lw=1.8, color="#334155"),
    )
    ax.text(0.5, 0.535, "Comparison", ha="center", va="bottom", fontsize=9.5, color="#334155")

    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
