from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(path, size=384):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def deterministic_preprocess(path, size=384):
    arr = load_rgb_image(path, size=size)
    return arr.transpose(2, 0, 1)
