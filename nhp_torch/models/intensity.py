from __future__ import annotations

import torch
from torch import nn


class IntensityHead(nn.Module):
    """Map hidden state h(t) -> intensities λ_k(t) (positive)."""

    def __init__(self, hidden_dim: int, num_types: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_types)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., hidden_dim) -> (..., num_types)
        return self.softplus(self.linear(h)) + 1e-8


