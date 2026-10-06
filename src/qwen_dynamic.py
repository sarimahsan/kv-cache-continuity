"""
src/qwen_dynamic.py

Dynamic Forward Engine for Qwen-2.5 Models with Decoupled Teacher-Forced KV Ablation.
Supports:
  1. Decoupled ablation modes:
     - 'full': Standard 36-layer execution, pristine KV cache.
     - 'readout_only': Pristine KV attention cache, shallow readout from exit depth d_t.
     - 'attention_only': Terminal 36-layer readout, past tokens attend to reconstructed KV cache.
     - 'combined': Shallow readout from d_t + reconstructed KV cache (real-world early exit).
  2. KV reconstruction strategies:
     - 'native_projection': Projects exit representation h_{d_t} via W_K^{(l)} LN_1^{(l)}(h_{d_t}) and W_V^{(l)} LN_1^{(l)}(h_{d_t}).
     - 'direct_copy': Copies (K, V) from layer (d_t - 1) into upper layers.
     - 'oracle_full': Uses true full-depth KV representations.
  3. Per-layer Cosine Similarity and Relative Frobenius Error tracking for (K, V).
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embeddings to key or query tensor x."""
    return (x * cos) + (rotate_half(x) * sin)


class QwenDynamicEngine:
    """
    Execution engine for Qwen-2.5 architecture supporting step-by-step
    decoupled ablation and dynamic depth evaluation.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        candidate_layers: Optional[List[int]] = None,
        **kwargs,
    ):
        self.device = torch.device(device)
        if dtype is None:
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif self.device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype
        self.candidate_layers = candidate_layers

        print(f"[*] Loading tokenizer for '{model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        print(f"[*] Loading pretrained causal model '{model_id}'...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.num_heads)
        self.hidden_size = self.config.hidden_size

    def init_static_kv_cache(self, batch_size: int = 1, max_seq_len: int = 512) -> Dict[str, List[torch.Tensor]]:
        """Pre-allocates contiguous static KV buffers for all layers."""
        k_buffers = [
            torch.zeros(
                batch_size, self.num_kv_heads, max_seq_len, self.head_dim,
                dtype=self.dtype, device=self.device
            )
            for _ in range(self.num_layers)
        ]
        v_buffers = [
            torch.zeros(
                batch_size, self.num_kv_heads, max_seq_len, self.head_dim,
                dtype=self.dtype, device=self.device
            )
            for _ in range(self.num_layers)
        ]
        return {"k": k_buffers, "v": v_buffers}

    def _get_cos_sin_single(self, layer, pos: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieves rotary cosine and sine embeddings for a single position index."""
        pos_tensor = torch.tensor([[pos]], dtype=torch.long, device=self.device)
        dummy_x = torch.zeros(1, 1, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        rotary_emb = getattr(layer.self_attn, "rotary_emb", None)
        if rotary_emb is None:
            rotary_emb = getattr(self.model.model, "rotary_emb", None)
        if rotary_emb is not None:
            return rotary_emb(dummy_x, pos_tensor)
        return None, None

    @torch.no_grad()
    def step_forward_decoupled(
        self,
        input_id: torch.Tensor,
        pos: int,
        kv_cache: Dict[str, List[torch.Tensor]],
        exit_depth: int,
        mode: str,
        strategy: str,
        track_cossim: bool = False,
        adapter: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[int, Dict[str, float]]]]:
        """
        Executes a single-token forward step under the specified decoupled experimental condition.

        Args:
            input_id: (B, 1) current token tensor.
            pos: Current sequence position index (0-indexed).
            kv_cache: Static KV buffer dict containing 'k' and 'v' lists.
            exit_depth: Exit depth d_t (1-indexed, e.g. 18, 24, 29).
            mode: 'full', 'readout_only', 'attention_only', or 'combined'.
            strategy: 'oracle_full', 'native_projection', or 'direct_copy'.
            track_cossim: If True, measures cosine similarity between true and projected (K, V).
            adapter: Optional ExitCalibrationModule for intermediate readout calibration.

        Returns:
            Tuple of (logits [B, 1, vocab_size], stats dict {layer: {'cos_k', 'cos_v', 'rel_k', 'rel_v'}} or None).
        """
        B = input_id.shape[0]
        h = self.model.model.embed_tokens(input_id)
        model_type = getattr(self.config, "model_type", "qwen2")
        if model_type in ["gemma", "gemma2"]:
            h = h * (self.config.hidden_size ** 0.5)

        stats = {} if track_cossim else None
        h_exit = None

        # Determine how many layers we physically execute in the forward backbone
        # In 'combined', we execute only 0..exit_depth-1
        # In 'full', 'readout_only', and 'attention_only', we execute all 36 layers
        exec_layers = exit_depth if mode == "combined" else self.num_layers

        for l in range(exec_layers):
            layer = self.model.model.layers[l]
            residual = h
            norm_h = layer.input_layernorm(h)

            attn = layer.self_attn
            q = attn.q_proj(norm_h).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(norm_h).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(norm_h).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

            cos, sin = self._get_cos_sin_single(layer, pos)
            if cos is not None and sin is not None:
                q = apply_rotary_pos_emb_single(q, cos, sin)
                k = apply_rotary_pos_emb_single(k, cos, sin)

            # Store computed true KV into cache
            # In attention_only with early exit: for l >= exit_depth, past KV was reconstructed
            if mode == "attention_only" and l >= exit_depth:
                # We save true (k, v) to compare with projected
                true_k = k.clone()
                true_v = v.clone()

                if strategy == "native_projection":
                    # Project from h_exit
                    norm_exit = layer.input_layernorm(h_exit)
                    k_proj = attn.k_proj(norm_exit).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    v_proj = attn.v_proj(norm_exit).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    if cos is not None and sin is not None:
                        k_proj = apply_rotary_pos_emb_single(k_proj, cos, sin)

                    kv_cache["k"][l][:, :, pos : pos + 1, :] = k_proj
                    kv_cache["v"][l][:, :, pos : pos + 1, :] = v_proj

                    if track_cossim and stats is not None:
                        cos_k = F.cosine_similarity(true_k.flatten().float(), k_proj.flatten().float(), dim=0).item()
                        cos_v = F.cosine_similarity(true_v.flatten().float(), v_proj.flatten().float(), dim=0).item()
                        rel_k = (torch.norm(true_k.float() - k_proj.float()) / (torch.norm(true_k.float()) + 1e-8)).item()
                        rel_v = (torch.norm(true_v.float() - v_proj.float()) / (torch.norm(true_v.float()) + 1e-8)).item()
                        stats[l] = {"cos_k": cos_k, "cos_v": cos_v, "rel_k": rel_k, "rel_v": rel_v}

                elif strategy == "oracle_k_proj_v":
                    # Experiment C: Oracle K + Projected V
                    norm_exit = layer.input_layernorm(h_exit)
                    v_proj = attn.v_proj(norm_exit).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    kv_cache["k"][l][:, :, pos : pos + 1, :] = k
                    kv_cache["v"][l][:, :, pos : pos + 1, :] = v_proj

                elif strategy == "proj_k_oracle_v":
                    # Experiment C: Projected K + Oracle V
                    norm_exit = layer.input_layernorm(h_exit)
                    k_proj = attn.k_proj(norm_exit).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    if cos is not None and sin is not None:
                        k_proj = apply_rotary_pos_emb_single(k_proj, cos, sin)
                    kv_cache["k"][l][:, :, pos : pos + 1, :] = k_proj
                    kv_cache["v"][l][:, :, pos : pos + 1, :] = v

                elif strategy == "direct_copy":
                    src_k = kv_cache["k"][exit_depth - 1][:, :, pos : pos + 1, :].clone()
                    src_v = kv_cache["v"][exit_depth - 1][:, :, pos : pos + 1, :].clone()
                    kv_cache["k"][l][:, :, pos : pos + 1, :] = src_k
                    kv_cache["v"][l][:, :, pos : pos + 1, :] = src_v

                else:
                    kv_cache["k"][l][:, :, pos : pos + 1, :] = k
                    kv_cache["v"][l][:, :, pos : pos + 1, :] = v
            else:
                kv_cache["k"][l][:, :, pos : pos + 1, :] = k
                kv_cache["v"][l][:, :, pos : pos + 1, :] = v

            # Attend over cache history
            k_hist = kv_cache["k"][l][:, :, : pos + 1, :]
            v_hist = kv_cache["v"][l][:, :, : pos + 1, :]

            if self.num_kv_heads != self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k_hist = k_hist.repeat_interleave(n_rep, dim=1)
                v_hist = v_hist.repeat_interleave(n_rep, dim=1)

            attn_out = F.scaled_dot_product_attention(q, k_hist, v_hist, is_causal=False)
            attn_dim = self.num_heads * self.head_dim
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, attn_dim)

            if hasattr(layer, "pre_feedforward_layernorm"):
                # Gemma 2 dual layernorm structure
                attn_proj = attn.o_proj(attn_out)
                if hasattr(layer, "post_attention_layernorm") and layer.post_attention_layernorm is not None:
                    attn_proj = layer.post_attention_layernorm(attn_proj)
                h = residual + attn_proj
                mlp_in = layer.pre_feedforward_layernorm(h)
                mlp_out = layer.mlp(mlp_in)
                if hasattr(layer, "post_feedforward_layernorm") and layer.post_feedforward_layernorm is not None:
                    mlp_out = layer.post_feedforward_layernorm(mlp_out)
                h = h + mlp_out
            else:
                # Standard Llama / Qwen / Gemma 1 structure
                h = residual + attn.o_proj(attn_out)
                h = h + layer.mlp(layer.post_attention_layernorm(h))

            if l == exit_depth - 1:
                h_exit = h.clone()

        # If mode == 'combined', execute skipped layer KV state propagation
        if mode == "combined" and exit_depth < self.num_layers:
            for l in range(exit_depth, self.num_layers):
                layer = self.model.model.layers[l]
                cos, sin = self._get_cos_sin_single(layer, pos)

                if strategy == "native_projection":
                    norm_exit = layer.input_layernorm(h_exit)
                    k_proj = layer.self_attn.k_proj(norm_exit).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    v_proj = layer.self_attn.v_proj(norm_exit).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    if cos is not None and sin is not None:
                        k_proj = apply_rotary_pos_emb_single(k_proj, cos, sin)

                    kv_cache["k"][l][:, :, pos : pos + 1, :] = k_proj
                    kv_cache["v"][l][:, :, pos : pos + 1, :] = v_proj

                elif strategy == "direct_copy":
                    src_k = kv_cache["k"][exit_depth - 1][:, :, pos : pos + 1, :].clone()
                    src_v = kv_cache["v"][exit_depth - 1][:, :, pos : pos + 1, :].clone()
                    kv_cache["k"][l][:, :, pos : pos + 1, :] = src_k
                    kv_cache["v"][l][:, :, pos : pos + 1, :] = src_v

        # Determine readout state
        if mode in ("readout_only", "combined"):
            readout_h = h_exit
            if adapter is not None and exit_depth in getattr(adapter, "candidate_depths", []):
                readout_h = adapter.forward_layer(readout_h, depth=exit_depth)
        else:
            readout_h = h

        norm_readout = self.model.model.norm(readout_h)
        logits = self.model.lm_head(norm_readout)

        final_softcapping = getattr(self.config, "final_logit_softcapping", None)
        if final_softcapping is not None:
            logits = torch.tanh(logits / final_softcapping) * final_softcapping

        return logits, stats

    # Convenient method aliases for evaluation runners
    create_static_kv_buffer = init_static_kv_cache
    forward_step = step_forward_decoupled

