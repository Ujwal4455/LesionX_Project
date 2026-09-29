from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class LesionDataset(Dataset):
    def __init__(self, df, image_col, label_col=None, transform=None, image_size=384):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row[self.image_col]
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = image.transpose(2, 0, 1)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        sample = {"image": torch.tensor(image, dtype=torch.float32), "index": idx}
        if self.label_col is not None:
            target = row.get(self.label_col)
            sample["target"] = torch.tensor(float(target), dtype=torch.float32)
        return sample
