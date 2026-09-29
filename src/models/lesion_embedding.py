from __future__ import annotations

import numpy as np
import torch


def generate_lesion_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            emb = model(x)
            all_embeddings.append(emb.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 0), dtype=np.float32)
