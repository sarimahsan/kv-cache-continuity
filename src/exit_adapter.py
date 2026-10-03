"""
src/exit_adapter.py

Exit-Head Calibration Module (Tuned Lens / Low-Rank Manifold Adapter).

Mathematical Foundation:
Intermediate hidden representations h_l in layer l reside on an intermediate
manifold M_l that is not calibrated for the terminal unembedding matrix W_unembed.
This module provides a lightweight, parameter-efficient affine or residual mapping:
    P_exit^(l): M_l -> M_L
such that:
    h_calibrated = P_exit^(l)(h_l)
    logits_l = LMHead( LN_f( h_calibrated ) )

Features:
1. Supports Linear Tuned Lens (LayerNorm + Linear) and Low-Rank Residual (LoRA-style).
2. Base model parameters remain 100% frozen.
3. Training requires backprop ONLY through the adapter parameters (<0.5% model size),
   making calibration extremely fast and VRAM-efficient.
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleExitAdapter(nn.Module):
    """
    Lightweight calibration adapter for a single intermediate layer l.
    Supports:
      - 'linear': Tuned Lens style: LN(x) @ W + b
      - 'lora': Low-Rank Residual: x + (alpha/rank) * (LN(x) @ W_down @ W_up)
    """
    def __init__(
        self,
        d_model: int,
        adapter_type: str = "linear",
        rank: int = 128,
        alpha: float = 1.0,
        dropout: float = 0.0,
        use_norm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.adapter_type = adapter_type.lower()
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 1.0

        self.ln = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if self.adapter_type == "linear":
            self.linear = nn.Linear(d_model, d_model, bias=True)
            # Initialize close to identity
            nn.init.eye_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        elif self.adapter_type == "lora":
            self.down = nn.Linear(d_model, rank, bias=False)
            self.up = nn.Linear(rank, d_model, bias=False)
            # Standard LoRA zero-init on up-projection: strictly starts as identity at step 0
            nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.up.weight)

        elif self.adapter_type == "mlp":
            hidden_dim = rank if rank > 0 else (d_model * 2)
            self.fc1 = nn.Linear(d_model, hidden_dim, bias=True)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_dim, d_model, bias=True)
            # Zero-init on output projection so adapter begins strictly as identity
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            raise ValueError(f"Unknown adapter_type: '{adapter_type}'. Use 'linear', 'mlp', or 'lora'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps intermediate hidden state x in M_l to calibrated state in M_L.
        Shape: (B, T, d_model) -> (B, T, d_model)
        """
        orig_dtype = x.dtype
        if hasattr(self.ln, "weight") and self.ln.weight is not None:
            target_dtype = self.ln.weight.dtype
        elif self.adapter_type == "linear":
            target_dtype = self.linear.weight.dtype
        elif self.adapter_type == "mlp":
            target_dtype = self.fc1.weight.dtype
        else:
            target_dtype = self.down.weight.dtype

        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        norm_x = self.ln(x)
        if self.adapter_type == "linear":
            out = self.linear(norm_x)
        elif self.adapter_type == "mlp":
            delta = self.drop(self.fc2(self.act(self.fc1(norm_x))))
            out = x + delta
        else:
            delta = self.scaling * self.up(self.drop(self.down(norm_x)))
            out = x + delta
        return out.to(orig_dtype)




class ExitCalibrationModule(nn.Module):
    """
    Container holding exit-head adapters for a set of candidate exit depths.
    Example: {18: SingleExitAdapter, 24: SingleExitAdapter, 29: SingleExitAdapter}
    """
    def __init__(
        self,
        d_model: int,
        candidate_depths: List[int],
        adapter_type: str = "linear",
        rank: int = 128,
        bottleneck_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.candidate_depths = sorted(candidate_depths)
        self.adapter_type = adapter_type
        if bottleneck_dim is not None:
            rank = bottleneck_dim
        self.rank = rank

        self.adapters = nn.ModuleDict({
            str(d): SingleExitAdapter(d_model, adapter_type=adapter_type, rank=rank)
            for d in self.candidate_depths
        })

    def forward_layer(self, h_l: torch.Tensor, depth: int) -> torch.Tensor:
        """Calibrates hidden state h_l at depth d."""
        key = str(depth)
        if key in self.adapters:
            return self.adapters[key](h_l)
        return h_l

    def forward(self, h_l: torch.Tensor, depth: Optional[int] = None) -> torch.Tensor:
        """Alias for forward_layer to support standard Module.__call__."""
        if depth is not None:
            return self.forward_layer(h_l, depth)
        raise ValueError("depth must be specified when calling ExitCalibrationModule.")

    def save_checkpoint(self, path: str):
        """Saves adapter weights and configuration metadata."""
        meta = {
            "d_model": self.d_model,
            "candidate_depths": self.candidate_depths,
            "adapter_type": self.adapter_type,
            "rank": self.rank,
            "state_dict": self.state_dict()
        }
        torch.save(meta, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Union[str, torch.device] = "cpu") -> "ExitCalibrationModule":
        """Loads a saved ExitCalibrationModule."""
        meta = torch.load(path, map_location=device)
        module = cls(
            d_model=meta["d_model"],
            candidate_depths=meta["candidate_depths"],
            adapter_type=meta.get("adapter_type", "linear"),
            rank=meta.get("rank", 128)
        )
        module.load_state_dict(meta["state_dict"])
        module.to(device)
        return module
