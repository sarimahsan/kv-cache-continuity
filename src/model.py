"""
src/model.py

Dense Transformer Language Model with support for Dynamic Early-Exit,
Decoupled KV-Cache Ablation, and Multi-Head Latent Attention (MLA).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.0


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert self.d_model % self.n_heads == 0

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Any] = None,
        layer_idx: Optional[int] = None,
        pos: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None and layer_idx is not None:
            if pos is not None:
                kv_cache.update(layer_idx, k, v, pos)
                k_hist, v_hist = kv_cache.get_context(layer_idx, pos + 1)
                attn = F.scaled_dot_product_attention(q, k_hist, v_hist, is_causal=False)
            else:
                for t in range(T):
                    kv_cache.update(layer_idx, k[:, :, t:t+1, :], v[:, :, t:t+1, :], t)
                k_hist, v_hist = kv_cache.get_context(layer_idx, T)
                attn = F.scaled_dot_product_attention(q, k_hist, v_hist, is_causal=True)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))

        out = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-style Multi-Head Latent Attention compression."""
    def __init__(self, d_model: int, n_heads: int, d_latent: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_latent = d_latent
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.down_proj = nn.Linear(d_model, d_latent, bias=False)
        self.up_k = nn.Linear(d_latent, d_model, bias=False)
        self.up_v = nn.Linear(d_latent, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mla_cache: Optional[Any] = None,
        pos: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        c = self.down_proj(x)

        if mla_cache is not None:
            if pos is not None:
                mla_cache.update(c, pos)
                c_hist = mla_cache.get_latent(end_pos=pos + 1)
            else:
                for t in range(T):
                    mla_cache.update(c[:, t:t+1, :], t)
                c_hist = mla_cache.get_latent(end_pos=T)
            k = self.up_k(c_hist).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.up_v(c_hist).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=(pos is None and T > 1))
        else:
            k = self.up_k(c).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.up_v(c).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))

        out = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), c


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        d_ff = config.d_ff if getattr(config, "d_ff", None) else 4 * config.d_model
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Any] = None,
        pos: Optional[int] = None,
    ) -> torch.Tensor:
        h = x + self.attn(self.ln_1(x), kv_cache=kv_cache, layer_idx=self.layer_idx, pos=pos)
        h = h + self.mlp(self.ln_2(h))
        return h


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config, l) for l in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        h = self.drop(self.token_embeddings(input_ids) + self.pos_embeddings(pos))
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.ln_f(h))

    def forward_step(
        self,
        input_id: torch.Tensor,
        pos: int,
        kv_cache: Any,
        exit_depth: int,
        state_prop_strategy: str = "native_projection",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = input_id.shape[0]
        pos_t = torch.tensor([[pos]], device=input_id.device)
        h = self.drop(self.token_embeddings(input_id) + self.pos_embeddings(pos_t))

        for l in range(exit_depth):
            h = self.layers[l](h, kv_cache=kv_cache, pos=pos)

        h_exit = h
        if exit_depth < self.config.n_layers:
            kv_cache.propagate_state(
                exit_depth=exit_depth,
                pos=pos,
                h_exit=h_exit,
                layers=self.layers,
                strategy=state_prop_strategy,
            )

        logits = self.lm_head(self.ln_f(h_exit))
        return logits, h_exit

    def generate_dynamic(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        depth_policy: List[int],
        state_prop_strategy: str = "native_projection",
    ) -> Dict[str, Any]:
        from src.kv_cache import StaticKVCache
        B, T = prompt_ids.shape
        head_dim = self.config.d_model // self.config.n_heads
        cache = StaticKVCache(
            n_layers=self.config.n_layers,
            batch_size=B,
            n_heads=self.config.n_heads,
            head_dim=head_dim,
            max_seq_len=T + max_new_tokens + 16,
            device=prompt_ids.device,
        )

        pos_range = torch.arange(0, T, device=prompt_ids.device).unsqueeze(0)
        h = self.drop(self.token_embeddings(prompt_ids) + self.pos_embeddings(pos_range))
        for layer in self.layers:
            h = layer(h, kv_cache=cache)

        curr_token = torch.argmax(self.lm_head(self.ln_f(h[:, -1:, :])), dim=-1)
        output_ids = [prompt_ids]
        executed_depths = []

        curr_pos = T
        for step in range(max_new_tokens):
            d = depth_policy[step % len(depth_policy)]
            executed_depths.append(d)
            output_ids.append(curr_token)
            if step < max_new_tokens - 1:
                logits, _ = self.forward_step(
                    input_id=curr_token,
                    pos=curr_pos,
                    kv_cache=cache,
                    exit_depth=d,
                    state_prop_strategy=state_prop_strategy,
                )
                curr_token = torch.argmax(logits, dim=-1)
                curr_pos += 1

        return {
            "output_ids": torch.cat(output_ids, dim=1),
            "depths": executed_depths,
        }
