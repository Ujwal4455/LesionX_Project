import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        return self.encoder(x)
