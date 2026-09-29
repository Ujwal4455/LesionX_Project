import torch
import torch.nn as nn


class LesionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)
