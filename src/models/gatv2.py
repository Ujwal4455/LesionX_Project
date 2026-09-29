import torch
import torch.nn as nn


class GATv2Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(out_dim * 2, 1)

    def forward(self, x, adj):
        projected = self.linear(x)
        outputs = []
        for i in range(x.size(0)):
            neighbors = adj[i].nonzero(as_tuple=False).squeeze(-1)
            if neighbors.numel() == 0:
                outputs.append(projected[i])
                continue
            pair = torch.cat([projected[i].unsqueeze(0).repeat(neighbors.size(0), 1), projected[neighbors]], dim=1)
            logits = self.attn(pair).squeeze(-1)
            weights = torch.softmax(logits, dim=0)
            agg = (weights.unsqueeze(-1) * projected[neighbors]).sum(dim=0)
            outputs.append(agg)
        return torch.stack(outputs)


class GATv2Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = GATv2Layer(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        return self.layer2(x)
