"""CLIP linear probe for foundation-model comparison."""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig


class CLIPLinearProbe(nn.Module):
    """Frozen CLIP backbone with a trainable linear classifier."""

    def __init__(self, num_classes=8):
        """Load CLIP ViT-L/14 and attach a linear probe."""
        super().__init__()
        self.clip = self._load_clip()
        for param in self.clip.parameters():
            param.requires_grad = False
        feature_dim = self._feature_dim()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.clip.eval()

    def _load_clip(self):
        """Load the pretrained CLIP model or fall back to a local config."""
        try:
            return CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        except Exception as exc:
            warnings.warn(
                f"Falling back to a randomly initialized CLIP configuration because pretrained weights could not be loaded: {exc}",
                RuntimeWarning,
            )
            text_config = CLIPTextConfig(
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
                max_position_embeddings=77,
                vocab_size=49408,
            )
            vision_config = CLIPVisionConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=24,
                num_attention_heads=16,
                image_size=224,
                patch_size=14,
            )
            config = CLIPConfig(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), projection_dim=768)
            return CLIPModel(config)

    def _feature_dim(self):
        """Infer the dimensionality of CLIP image features."""
        projection_dim = getattr(self.clip.config, "projection_dim", None)
        if projection_dim is not None:
            return int(projection_dim)
        vision_hidden = getattr(getattr(self.clip.config, "vision_config", None), "hidden_size", None)
        return int(vision_hidden or 1024)

    def forward(self, x):
        """Encode images with frozen CLIP features and classify them."""
        with torch.no_grad():
            features = self.clip.get_image_features(pixel_values=x)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.classifier(features)

    def train(self, mode=True):
        """Keep the frozen CLIP backbone in eval mode."""
        super().train(mode)
        self.clip.eval()
        return self

    @classmethod
    def clip_transforms(cls):
        """Return the CLIP-specific preprocessing pipeline."""
        return T.Compose(
            [
                T.Resize(224, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
