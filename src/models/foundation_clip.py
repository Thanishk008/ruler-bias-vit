"""CLIP-based foundation-model utilities."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from transformers import CLIPModel, CLIPTokenizer


CLASS_NAME_ALIASES = {
    "MEL": "melanoma",
    "NV": "melanocytic nevus",
    "BCC": "basal cell carcinoma",
    "AK": "actinic keratosis",
    "BKL": "benign keratosis",
    "DF": "dermatofibroma",
    "VASC": "vascular lesion",
    "SCC": "squamous cell carcinoma",
}

ZERO_SHOT_TEMPLATES = [
    "a dermoscopic image of {}",
    "a photo of {}",
    "a skin lesion showing {}",
]


class CLIPZeroShotClassifier(nn.Module):
    """Frozen CLIP backbone with zero-shot text prompts for inference."""

    def __init__(self, class_names, prompt_templates=None):
        """Load CLIP and precompute text embeddings for the target classes."""
        super().__init__()
        self.class_names = list(class_names)
        self.prompt_templates = list(prompt_templates or ZERO_SHOT_TEMPLATES)
        self.clip = self._load_clip()
        for param in self.clip.parameters():
            param.requires_grad = False
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.register_buffer("text_features", self._build_text_features(), persistent=False)
        self.clip.eval()

    def _load_clip(self):
        """Load the pretrained CLIP model used for zero-shot inference."""
        return CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    def _label_text(self, class_name: str) -> str:
        """Expand dataset abbreviations into prompt-friendly class names."""
        return CLASS_NAME_ALIASES.get(class_name, class_name.lower())

    def _build_text_features(self):
        """Precompute normalized text embeddings for each class prompt."""
        device = next(self.clip.parameters()).device
        class_features = []
        with torch.no_grad():
            for class_name in self.class_names:
                label_text = self._label_text(class_name)
                prompts = [template.format(label_text) for template in self.prompt_templates]
                tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt")
                tokenized = {key: value.to(device) for key, value in tokenized.items()}
                features = self.clip.get_text_features(**tokenized).pooler_output
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                features = features.mean(dim=0)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                class_features.append(features)
        return torch.stack(class_features, dim=0)

    def forward(self, x):
        """Run zero-shot CLIP inference with fixed class prompts."""
        with torch.no_grad():
            image_features = self.clip.get_image_features(pixel_values=x).pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        logit_scale = self.clip.logit_scale.exp()
        return logit_scale * image_features @ self.text_features.t()

    @classmethod
    def clip_transforms(cls):
        """Return the CLIP preprocessing pipeline used for zero-shot inference."""
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
