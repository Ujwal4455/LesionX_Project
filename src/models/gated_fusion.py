import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, visual_dim, metadata_dim=0, morphology_dim=0, tbp_dim=0, embedding_dim=256):
        super().__init__()
        feature_dims = [visual_dim]
        if metadata_dim > 0:
            feature_dims.append(metadata_dim)
        if morphology_dim > 0:
            feature_dims.append(morphology_dim)
        if tbp_dim > 0:
            feature_dims.append(tbp_dim)

        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid()) for dim in feature_dims])
        self.proj = nn.Linear(sum(feature_dims), embedding_dim)

    def forward(self, *features):
        if not features:
            raise ValueError("At least one feature tensor is required.")

        gated = []
        if len(features) != len(self.gates):
            raise ValueError(f"Expected {len(self.gates)} feature tensors, got {len(features)}")

        for feat, gate in zip(features, self.gates):
            if feat.dim() != 2:
                raise ValueError(f"Feature tensor must be 2D, got shape {tuple(feat.shape)}")
            gated.append(gate(feat) * feat)

        fused = torch.cat(gated, dim=1)
        return self.proj(fused)
