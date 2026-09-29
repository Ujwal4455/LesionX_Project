from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def assess_image_quality(image_path):
    if not image_path or not Path(image_path).exists():
        return {
            "exists": False,
            "laplacian_blur": None,
            "brightness": None,
            "contrast": None,
        }

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {
            "exists": False,
            "laplacian_blur": None,
            "brightness": None,
            "contrast": None,
        }

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    blur = float(laplacian.var())
    brightness = float(image.mean())
    contrast = float(image.std())

    return {
        "exists": True,
        "laplacian_blur": blur,
        "brightness": brightness,
        "contrast": contrast,
    }
