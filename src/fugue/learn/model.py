"""The counterpoint network: a sum of one learned vector per feature (an embedding bag), then two
small layers. Because the first layer is a sum of lookups it costs almost nothing in the browser."""
import torch, torch.nn as nn
from .grid import TABLES, NAMES, N_OUT

class Counterpoint(nn.Module):
    def __init__(self, hidden=128, drop=0.3):
        super().__init__()
        sizes = [TABLES[n] for n in NAMES]
        self.offsets = torch.tensor([0] + sizes[:-1]).cumsum(0)
        self.emb = nn.Embedding(sum(sizes), hidden)
        self.b0 = nn.Parameter(torch.zeros(hidden))
        self.l1 = nn.Linear(hidden, hidden); self.out = nn.Linear(hidden, N_OUT)
        self.drop = nn.Dropout(drop)
        nn.init.normal_(self.emb.weight, std=0.05)
    def forward(self, x):                      # x: [B, F] categorical indices
        h = self.emb(x.long() + self.offsets).sum(1) + self.b0
        h = self.drop(torch.relu(h))
        h = self.drop(torch.relu(self.l1(h)))
        return self.out(h)
