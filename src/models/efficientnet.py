from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class EfficientNetV2SFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()
        if pretrained:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
            backbone = models.efficientnet_v2_s(weights=weights)
        else:
            backbone = models.efficientnet_v2_s(weights=None)

        self.backbone = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = nn.Linear(backbone.classifier[1].in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
