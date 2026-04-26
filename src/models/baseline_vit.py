"""Baseline ViT model that preserves the standard ruler-bias failure mode."""

from __future__ import annotations

from typing import List

import timm
import torch
from torch import nn


class BaselineViT(nn.Module):
    """Standard ViT-B/16 classification model with attention introspection."""

    def __init__(self, num_classes=8, pretrained=True):
        """Build the baseline ViT model with an 8-class head."""
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        """Run a forward pass and return logits."""
        return self.model(x)

    def get_attention_weights(self, x) -> List[torch.Tensor]:
        """Capture per-block attention maps from the final ViT blocks."""
        attention_weights: List[torch.Tensor] = []
        hooks = []

        def _capture_attention(_, __, output):
            attention_weights.append(output)

        for block in self.model.blocks:
            hooks.append(block.attn.attn_drop.register_forward_hook(_capture_attention))

        try:
            _ = self.model(x)
        finally:
            for hook in hooks:
                hook.remove()

        return attention_weights
