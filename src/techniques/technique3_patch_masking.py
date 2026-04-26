"""Border-patch masking hook used as an architectural debiasing technique."""

from __future__ import annotations

import math

import torch

from .technique2_attention_reg import BORDER_PATCH_INDICES


class BorderPatchMasker:
    """Mask border patch embeddings during training via a forward hook."""

    def __init__(self, model, mask_prob=0.5):
        """Attach a masking hook to the model patch embedding layer."""
        self.model = model
        self.mask_prob = mask_prob
        patch_embed = getattr(self.model.model, "patch_embed", None)
        if patch_embed is None:
            raise AttributeError("Could not find patch_embed on the wrapped model.")
        self.hook_handle = patch_embed.register_forward_hook(self._hook)

    def _resolve_border_indices(self, token_count: int):
        """Resolve the border patch indices for the current token grid."""
        if token_count == len(BORDER_PATCH_INDICES):
            return BORDER_PATCH_INDICES
        grid_size = int(math.isqrt(token_count))
        if grid_size * grid_size != token_count:
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

    def _mask_output(self, output: torch.Tensor):
        """Apply Bernoulli masking to border patch embeddings."""
        if not self.model.training or self.mask_prob <= 0 or output.ndim != 3:
            return output
        batch_size, token_count, _ = output.shape
        indices = self._resolve_border_indices(token_count)
        if not indices:
            return output
        keep_prob = 1.0 - self.mask_prob
        mask = torch.bernoulli(torch.full((batch_size, len(indices)), keep_prob, device=output.device)).to(
            output.dtype
        )
        masked = output.clone()
        masked[:, indices, :] = masked[:, indices, :] * mask.unsqueeze(-1)
        return masked

    def _hook(self, module, inputs, output):
        """Forward hook that masks the patch embeddings only in training mode."""
        if isinstance(output, torch.Tensor):
            return self._mask_output(output)
        return output

    def remove_hook(self):
        """Remove the registered forward hook."""
        self.hook_handle.remove()


def wrap_model_with_masker(model, mask_prob=0.5):
    """Attach a border patch masker and return both the model and masker."""
    masker = BorderPatchMasker(model, mask_prob=mask_prob)
    return model, masker
