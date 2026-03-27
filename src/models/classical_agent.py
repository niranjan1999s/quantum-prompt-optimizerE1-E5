from __future__ import annotations

import torch
from torch import nn


class ClassicalGoliathAgent(nn.Module):
    """A large-capacity classical baseline. Hidden sizes are fixed at 256/512
    so Goliath always retains a massive parameter advantage regardless of qubit count."""
    def __init__(self, n_inputs: int = 8, embed_dim: int = 768) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ClassicalDavidAgent(nn.Module):
    def __init__(self, n_inputs: int = 4, hidden_size: int = 4, embed_dim: int = 768, exact_params: int | None = 4080) -> None:
        super().__init__()
        # A minimal 3-layer MLP that can fit in tiny budgets (e.g., 2k-5k params)
        self.network = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim),
        )
        
        current_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        if exact_params is not None:
            if exact_params > current_params:
                # Add a dummy parameter to perfectly match the VQC parameter count down to the exact digit
                self.dummy_param = nn.Parameter(torch.zeros(exact_params - current_params))
            elif exact_params < current_params:
                raise ValueError(f"exact_params={exact_params} is too small for hidden_size={hidden_size} which requires {current_params}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if hasattr(self, 'dummy_param'):
            # The dummy param contributes to param count but doesn't change the function expressivity
            out = out + (self.dummy_param.sum() * 0.0)
        return out

