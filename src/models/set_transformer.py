import torch
import torch.nn as nn


class SetTransformer(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, attention_heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList()
        for _ in range(layers):
            attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=attention_heads, batch_first=True, dropout=dropout)
            self.layers.append(attn)
            self.layers.append(nn.LayerNorm(input_dim))
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.Linear(hidden_dim, input_dim))

    def forward(self, x, mask=None):
        out = x
        for layer in self.layers:
            if isinstance(layer, nn.MultiheadAttention):
                attn_out, _ = layer(out, out, out, key_padding_mask=mask)
                out = out + attn_out
            elif isinstance(layer, nn.LayerNorm):
                out = layer(out)
            elif isinstance(layer, nn.Linear):
                out = layer(out)
            elif isinstance(layer, nn.ReLU):
                out = layer(out)
            elif isinstance(layer, nn.Dropout):
                out = layer(out)
        return out
