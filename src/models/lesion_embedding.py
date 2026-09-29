from __future__ import annotations

import numpy as np
import torch


def generate_lesion_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            emb = model(x)
            embeddings.append(emb.cpu().numpy())
    if len(embeddings) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(embeddings, axis=0)
