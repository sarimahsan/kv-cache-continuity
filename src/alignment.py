import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class IdentityAlignment(nn.Module):
    """
    Rank 0 / Identity Alignment:
      P(x) = x
      P^-(x) = x
    Represents Naive Layer Substitution with zero trainable parameters.
    """
    def __init__(self):
        super().__init__()

    def align_in(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def align_out(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def count_parameters(self) -> int:
        return 0


class TiedOppositeAlignmentPair(nn.Module):
    """
    Tied Opposite Residual Transform:
      P(x)   = x + (alpha / rank) * (x @ W_down @ W_up)
      P^-(x) = x - (alpha / rank) * (x @ W_down @ W_up)
      
    Shares W_down and W_up between input and output transforms.
    Zero-initialized up-projection so it strictly starts as Identity at step 0.
    """
    def __init__(self, d_model: int, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.scaling = alpha / rank

        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def align_in(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scaling * self.up(self.down(x))

    def align_out(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.scaling * self.up(self.down(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UntiedAlignmentPair(nn.Module):
    """
    Untied Independent Low-Rank Alignment:
      P_in(x)  = x + (alpha / rank) * (x @ W_in_down @ W_in_up)
      P_out(x) = x + (alpha / rank) * (x @ W_out_down @ W_out_up)
    """
    def __init__(self, d_model: int, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.scaling = alpha / rank

        self.in_down = nn.Linear(d_model, rank, bias=False)
        self.in_up = nn.Linear(rank, d_model, bias=False)
        self.out_down = nn.Linear(d_model, rank, bias=False)
        self.out_up = nn.Linear(rank, d_model, bias=False)

        nn.init.normal_(self.in_down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.in_up.weight)
        nn.init.normal_(self.out_down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_up.weight)

    def align_in(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scaling * self.in_up(self.in_down(x))

    def align_out(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scaling * self.out_up(self.out_down(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AlignedBlockWrapper(nn.Module):
    """
    Chains alignment pair around a target computational block:
      h_{l+1} = P_out( F( P_in( h_l ) ) )
    """
    def __init__(self, target_layer: nn.Module, adapter_pair: nn.Module):
        super().__init__()
        self.target_layer = target_layer
        self.adapter = adapter_pair

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_aligned = self.adapter.align_in(x)
        out = self.target_layer(x_aligned)
        out_realigned = self.adapter.align_out(out)
        return out_realigned

    def count_trainable_parameters(self) -> int:
        return self.adapter.count_parameters()


class ParameterMatchedAdapter(nn.Module):
    """
    Parameter-Matched Residual Control:
    A bottleneck residual adapter with the exact same parameter budget
    as the alignment pair, executed directly without any layer F.
    """
    def __init__(self, d_model: int, total_param_budget: int):
        super().__init__()
        bottleneck_dim = max(1, total_param_budget // (2 * d_model))
        self.fc1 = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(bottleneck_dim, d_model, bias=False)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
