"""
src/router.py

Dynamic Exit Router for Bucketed CUDA Graph Inference.
Maps intermediate transformer hidden states to discrete execution buckets (e.g. {4, 6, 8, 10, 12}).

Supports:
1. Sync-free device-level bucket selection (no .item(), no host-device syncs).
2. Confidence-based early exit and classification-based bucket selection.
3. Lightweight MLP and linear router architectures.
4. Oracle label generation and confusion matrix evaluation.
"""

from typing import List, Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class BucketRouter(nn.Module):
    """
    Lightweight router operating on intermediate representation h_probe
    to select the optimal execution bucket from a predefined set of depths.
    """

    def __init__(
        self,
        d_model: int = 768,
        buckets: Optional[List[int]] = None,
        probe_depth: int = 4,
        hidden_dim: Optional[int] = None,
        confidence_threshold: float = 0.85,
    ):
        super().__init__()
        if buckets is None:
            buckets = [4, 6, 8, 10, 12]
        self.buckets = sorted(list(set(buckets)))
        self.num_buckets = len(self.buckets)
        self.probe_depth = probe_depth
        self.confidence_threshold = confidence_threshold

        # Map bucket depth to class index and vice versa
        self.depth_to_idx = {d: i for i, d in enumerate(self.buckets)}
        self.idx_to_depth = {i: d for i, d in enumerate(self.buckets)}

        # Device tensor containing bucket depths for sync-free indexing
        self.register_buffer("bucket_tensor", torch.tensor(self.buckets, dtype=torch.long))

        if hidden_dim is not None and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_buckets, bias=True),
            )
        else:
            self.net = nn.Linear(d_model, self.num_buckets, bias=True)

    def forward(self, h_probe: torch.Tensor) -> torch.Tensor:
        """
        Computes router logits.
        Args:
            h_probe: (B, 1, d_model) or (B, d_model)
        Returns:
            logits: (B, num_buckets)
        """
        if h_probe.dim() == 3:
            h_probe = h_probe.squeeze(1)
        return self.net(h_probe)

    def predict_bucket_sync_free(self, h_probe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Completely SYNC-FREE bucket selection entirely on GPU device:
        Does NOT invoke .item(), .cpu(), or any synchronization.

        Returns:
            selected_bucket_tensor: (B,) int64 tensor containing the chosen depth in self.buckets
            bucket_idx_tensor: (B,) int64 tensor containing 0..num_buckets-1
        """
        logits = self.forward(h_probe)
        bucket_idx = torch.argmax(logits, dim=-1)
        selected_bucket = self.bucket_tensor[bucket_idx]
        return selected_bucket, bucket_idx

    def predict_bucket_eager(self, h_probe: torch.Tensor) -> int:
        """
        Eager evaluation returning Python int (for debugging / eager baseline comparison).
        Warning: Invokes .item() which triggers host-device sync!
        """
        selected_bucket, _ = self.predict_bucket_sync_free(h_probe)
        return int(selected_bucket[0].item())

    @torch.no_grad()
    def calibrate_from_oracle(
        self,
        probe_states: torch.Tensor,
        oracle_best_buckets: torch.Tensor,
        lr: float = 1e-3,
        steps: int = 200,
    ):
        """
        Trains or calibrates router weights against oracle optimal bucket selections.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        self.train()

        for _ in range(steps):
            optimizer.zero_grad()
            logits = self.forward(probe_states)
            loss = criterion(logits, oracle_best_buckets)
            loss.backward()
            optimizer.step()
        self.eval()
