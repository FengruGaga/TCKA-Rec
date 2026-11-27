"""Conditional low-rank readout over a frozen backbone."""

import torch
from torch import nn


class ConditionalReadout(nn.Module):
    """Task-private low-rank conditional interaction over frozen h and u."""

    def __init__(self, rdim, rank=4):
        super().__init__()
        self.h_norm = nn.LayerNorm(rdim, elementwise_affine=False)
        self.u_norm = nn.LayerNorm(rdim, elementwise_affine=False)
        self.A = nn.Linear(2 * rdim, rank, bias=False)
        self.C = nn.Linear(rdim, rank, bias=False)
        self.out = nn.Linear(rank, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def delta(self, h, u):
        h, u = self.h_norm(h.detach()), self.u_norm(u.detach())
        q = self.A(torch.cat([h, u], dim=1)).sigmoid()
        return self.out(q * self.C(u)).reshape(-1)

    def forward(self, h, u, p):
        p = p.detach().reshape(-1)
        logits = torch.logit(p.clamp(1e-7, 1 - 1e-7))
        difference = (logits + self.delta(h, u)).sigmoid() - logits.sigmoid()
        return (p + difference).clamp(0, 1)


@torch.no_grad()
def old_task_hidden(base, batch, position):
    """Return the original pre-tower fusion while retaining the batch axis."""
    env = base.env_embedding_network.weight[position]
    experts = torch.stack([batch[f's{position}'] * env, batch['g']], dim=2)
    weights = base.gate_networks[position](batch['x'])
    return (experts @ weights.unsqueeze(2)).squeeze(-1)
