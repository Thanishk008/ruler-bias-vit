"""Data-level debiasing utilities for ruler suppression and augmentation."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw


def detect_ruler(image: Image.Image, border_frac=0.08, white_thresh=240, black_thresh=15, contamination_thresh=0.05) -> bool:
    """Detect ruler-like border contamination in a dermoscopic image."""
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    border_h = max(1, int(round(height * border_frac)))
    border_w = max(1, int(round(width * border_frac)))

    mask = np.zeros_like(gray, dtype=bool)
    mask[:border_h, :] = True
    mask[-border_h:, :] = True
    mask[:, :border_w] = True
    mask[:, -border_w:] = True

    border_pixels = gray[mask]
    if border_pixels.size == 0:
        return False
    contaminated = np.logical_or(border_pixels > white_thresh, border_pixels < black_thresh).sum()
    return (contaminated / float(border_pixels.size)) > contamination_thresh


class RulerCropTransform:
    """Crop border regions when ruler contamination is detected."""

    def __init__(self, border_frac=0.08):
        """Store the border fraction used for detection and cropping."""
        self.border_frac = border_frac

    def __call__(self, img: Image.Image):
        """Crop the border and resize the image if a ruler is detected."""
        if not detect_ruler(img, border_frac=self.border_frac):
            return img
        width, height = img.size
        border_w = max(1, int(round(width * self.border_frac)))
        border_h = max(1, int(round(height * self.border_frac)))
        left = border_w
        top = border_h
        right = max(left + 1, width - border_w)
        bottom = max(top + 1, height - border_h)
        cropped = img.crop((left, top, right, bottom))
        return cropped.resize((224, 224), Image.Resampling.BILINEAR)


class SyntheticRulerAugmentation:
    """Inject synthetic ruler-like lines near image borders."""

    def __init__(self, p=0.5, border_frac=0.08):
        """Store the augmentation probability and border extent."""
        self.p = p
        self.border_frac = border_frac

    def _border_band(self, width: int, height: int) -> Tuple[int, int]:
        """Compute a border band width in pixels."""
        return max(1, int(round(width * self.border_frac))), max(1, int(round(height * self.border_frac)))

    def __call__(self, img: Image.Image):
        """Draw random synthetic border markings with probability p."""
        if random.random() > self.p:
            return img

        image = img.convert("RGB").copy()
        draw = ImageDraw.Draw(image)
        width, height = image.size
        border_w, border_h = self._border_band(width, height)
        line_count = random.randint(1, 3)

        for _ in range(line_count):
            color = random.choice([(255, 255, 255), (0, 0, 0)])
            thickness = random.randint(1, 3)
            orientation = random.choice(["horizontal", "vertical", "diagonal"])
            if orientation == "horizontal":
                y = random.choice(
                    [
                        random.randint(0, border_h - 1),
                        random.randint(max(0, height - border_h), height - 1),
                    ]
                )
                draw.line((0, y, width - 1, y), fill=color, width=thickness)
            elif orientation == "vertical":
                x = random.choice(
                    [
                        random.randint(0, border_w - 1),
                        random.randint(max(0, width - border_w), width - 1),
                    ]
                )
                draw.line((x, 0, x, height - 1), fill=color, width=thickness)
            else:
                corners = [
                    (random.randint(0, border_w - 1), random.randint(0, border_h - 1)),
                    (random.randint(max(0, width - border_w), width - 1), random.randint(0, border_h - 1)),
                    (random.randint(0, border_w - 1), random.randint(max(0, height - border_h), height - 1)),
                    (
                        random.randint(max(0, width - border_w), width - 1),
                        random.randint(max(0, height - border_h), height - 1),
                    ),
                ]
                start, end = random.sample(corners, 2)
                draw.line((*start, *end), fill=color, width=thickness)

        return image
