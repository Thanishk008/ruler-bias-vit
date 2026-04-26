"""Attention regularization that suppresses border-patch focus."""

from __future__ import annotations

import math

import torch
from torch import nn


BORDER_PATCH_INDICES = sorted(
    set(
        list(range(14))
        + list(range(182, 196))
        + list(range(0, 196, 14))
        + list(range(13, 196, 14))
    )
)


def _border_patch_indices(token_count: int):
    """Return border indices for a square patch grid, excluding any CLS token."""
    if token_count <= 0:
        return []

    patch_count = token_count
    if token_count - 1 > 0:
        cls_compatible = int(math.isqrt(token_count - 1))
        if cls_compatible * cls_compatible == token_count - 1:
            patch_count = token_count - 1

    grid_size = int(math.isqrt(patch_count))
    if grid_size * grid_size != patch_count:
        return []

    border_width = max(1, int(round(grid_size * 0.08)))
    indices = []
    for row in range(grid_size):
        for col in range(grid_size):
            if (
                row < border_width
                or row >= grid_size - border_width
                or col < border_width
                or col >= grid_size - border_width
            ):
                indices.append(row * grid_size + col)
    return sorted(set(indices))


class AttentionRegularizationLoss(nn.Module):
    """Penalize CLS-token attention that falls on border patches."""

    def __init__(self, lambda_weight=0.1):
        """Store the regularization strength."""
        super().__init__()
        self.lambda_weight = lambda_weight

    def forward(self, attention_weights_list):
        """Compute the mean border-attention penalty over all blocks."""
        if not attention_weights_list:
            return torch.zeros(())

        penalties = []
        for attn in attention_weights_list:
            if attn.ndim != 4:
                continue
            token_count = attn.shape[-1]
            border_indices = _border_patch_indices(token_count)
            if not border_indices:
                continue

            if token_count - 1 > 0 and int(math.isqrt(token_count - 1)) ** 2 == token_count - 1:
                border_indices_offset = [i + 1 for i in border_indices]
                border_attn = attn[:, :, 0, border_indices_offset]
            else:
                border_attn = attn[:, :, :, border_indices]

            penalties.append(border_attn.mean())

        if not penalties:
            return torch.zeros(())

        return self.lambda_weight * torch.stack(penalties).mean()
