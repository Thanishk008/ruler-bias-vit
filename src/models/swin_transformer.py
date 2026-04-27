"""Swin Transformer model that improves robustness to border artifacts."""

from __future__ import annotations

from typing import List

import timm
import torch
from torch import nn


class SwinTransformer(nn.Module):
    """Swin-Tiny classifier built with timm."""

    def __init__(self, num_classes=8):
        """Build the Swin-Tiny classifier with an 8-class head."""
        super().__init__()
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=num_classes,
        )

    def forward(self, x):
        """Run a forward pass and return logits."""
        return self.model(x)

    def get_attention_weights(self, x) -> List[torch.Tensor]:
        """Capture per-block attention maps from the Swin hierarchy."""
        attention_weights: List[torch.Tensor] = []
        hooks = []

        def _capture_attention(_, __, output):
            attention_weights.append(output)

        for layer in self.model.layers:
            for block in layer.blocks:
                hooks.append(block.attn.attn_drop.register_forward_hook(_capture_attention))

        try:
            _ = self.model(x)
        finally:
            for hook in hooks:
                hook.remove()

        return attention_weights
