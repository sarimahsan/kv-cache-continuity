"""
tests/test_kv_continuity.py

Unit tests for StaticKVCache and Dynamic Early-Exit Forward Step.
Tests KV continuity, hole-filling (state propagation), and multi-step generation.
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.model import ModelConfig, TransformerLM, MultiHeadLatentAttention
from src.kv_cache import StaticKVCache, MLAKVCache



class TestKVContinuity(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.config = ModelConfig(
            vocab_size=100,
            n_layers=4,
            d_model=64,
            n_heads=2,
            d_ff=128,
            max_seq_len=64,
            dropout=0.0
        )
        self.model = TransformerLM(self.config)
        self.model.eval()

    def test_cache_initialization_and_reset(self):
        cache = StaticKVCache(
            n_layers=4,
            batch_size=2,
            n_heads=2,
            head_dim=32,
            max_seq_len=16
        )
        self.assertEqual(len(cache.k_cache), 4)
        self.assertEqual(len(cache.v_cache), 4)
        self.assertEqual(cache.k_cache[0].shape, (2, 2, 16, 32))

        # Insert some dummy values
        k_dummy = torch.ones(2, 2, 1, 32)
        v_dummy = torch.ones(2, 2, 1, 32)
        cache.update(0, k_dummy, v_dummy, pos=3)
        k_ret, v_ret = cache.get_slot(0, pos=3)
        self.assertTrue(torch.allclose(k_ret, k_dummy))

        # Reset
        cache.reset()
        k_ret_reset, _ = cache.get_slot(0, pos=3)
        self.assertTrue(torch.all(k_ret_reset == 0.0))

    def test_state_propagation_eliminates_holes(self):
        """
        Verify that early exit at depth d_t < L fills slots for layers >= d_t
        via native projection and direct copy without leaving zeros.
        """
        cache = StaticKVCache(
            n_layers=4,
            batch_size=1,
            n_heads=2,
            head_dim=32,
            max_seq_len=16
        )

        h_exit = torch.randn(1, 1, 64)
        pos = 2
        exit_depth = 2  # Executed layers 0 and 1; skipped layers 2 and 3

        # Run native projection
        cache.propagate_state(
            exit_depth=exit_depth,
            pos=pos,
            h_exit=h_exit,
            layers=self.model.layers,
            strategy="native_projection"
        )

        # Layers 2 and 3 must now be non-zero at position 2
        k2, v2 = cache.get_slot(2, pos)
        k3, v3 = cache.get_slot(3, pos)
        self.assertFalse(torch.all(k2 == 0.0), "Layer 2 K slot should be populated by state propagation")
        self.assertFalse(torch.all(v2 == 0.0), "Layer 2 V slot should be populated by state propagation")
        self.assertFalse(torch.all(k3 == 0.0), "Layer 3 K slot should be populated by state propagation")
        self.assertFalse(torch.all(v3 == 0.0), "Layer 3 V slot should be populated by state propagation")

    def test_variable_depth_decoding_continuity(self):
        """
        Simulate an adversarial variable-depth sequence:
          Step 0: Token exits at Layer 1 (shallowest)
          Step 1: Token needs Layer 4 (full depth - queries Layer 2,3,4 over Step 0!)
          Step 2: Token exits at Layer 2
          Step 3: Token needs Layer 4
        Must execute without crash, zero NaN, and correct output shapes.
        """
        B = 1
        prompt = torch.tensor([[10, 20, 30]])
        T_prompt = prompt.shape[1]

        kv_cache = StaticKVCache(
            n_layers=4,
            batch_size=B,
            n_heads=2,
            head_dim=32,
            max_seq_len=64
        )

        # Prefill prompt
        e_tok = self.model.token_embeddings(prompt)
        pos_range = torch.arange(0, T_prompt).unsqueeze(0)
        h = self.model.drop(e_tok + self.model.pos_embeddings(pos_range))
        for l in range(4):
            h = self.model.layers[l](h, kv_cache=kv_cache)

        # Variable depths for successive generated steps
        test_depths = [1, 4, 2, 4, 1, 3, 4]
        curr_token = torch.tensor([[5]])
        curr_pos = T_prompt

        for d in test_depths:
            logits, h_exit = self.model.forward_step(
                input_id=curr_token,
                pos=curr_pos,
                kv_cache=kv_cache,
                exit_depth=d,
                state_prop_strategy="native_projection"
            )
            self.assertEqual(logits.shape, (B, 1, 100))
            self.assertFalse(torch.isnan(logits).any(), f"NaN detected at step {curr_pos} with depth {d}")
            self.assertFalse(torch.isinf(logits).any(), f"Inf detected at step {curr_pos} with depth {d}")

            curr_token = torch.argmax(logits, dim=-1)
            curr_pos += 1

    def test_dynamic_generation_api(self):
        prompt = torch.tensor([[12, 15, 18]])
        result = self.model.generate_dynamic(
            prompt_ids=prompt,
            max_new_tokens=10,
            depth_policy=[2, 4, 1, 3],  # cyclic depth pattern
            state_prop_strategy="native_projection"
        )
        self.assertEqual(result["output_ids"].shape, (1, 13))
        self.assertEqual(len(result["depths"]), 10)
        self.assertEqual(result["depths"][:4], [2, 4, 1, 3])

    def test_mla_cache_and_memory_reduction(self):
        """
        Verify that MLAKVCache achieves massive memory compression vs standard MHA cache.
        """
        B = 2
        S_max = 512
        n_layers = 12
        n_heads = 12
        d_model = 768
        head_dim = 64
        d_latent = 128

        # Standard MHA cache memory
        mha_cache = StaticKVCache(
            n_layers=n_layers,
            batch_size=B,
            n_heads=n_heads,
            head_dim=head_dim,
            max_seq_len=S_max
        )
        mha_bytes = sum(k.numel() * k.element_size() for k in mha_cache.k_cache) + \
                    sum(v.numel() * v.element_size() for v in mha_cache.v_cache)

        # MLA unified latent cache memory
        mla_cache = MLAKVCache(
            batch_size=B,
            d_latent=d_latent,
            max_seq_len=S_max
        )
        mla_bytes = mla_cache.memory_footprint_bytes()

        compression_ratio = mha_bytes / mla_bytes
        self.assertGreater(compression_ratio, 50.0, f"Expected >50x compression, got {compression_ratio:.1f}x")

        # Test latent insertion and retrieval
        c_dummy = torch.randn(B, 1, d_latent)
        mla_cache.update(c_dummy, pos=5)
        ret = mla_cache.get_latent(end_pos=6)
        self.assertEqual(ret.shape, (B, 6, d_latent))
        self.assertTrue(torch.allclose(ret[:, 5:6, :], c_dummy))

    def test_mla_step_decoding(self):
        """
        Verify MultiHeadLatentAttention with MLAKVCache across prefill and incremental step decoding.
        """
        B = 1
        d_model = 64
        n_heads = 2
        d_latent = 16
        S_max = 32

        mla_layer = MultiHeadLatentAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_latent=d_latent
        )
        mla_cache = MLAKVCache(
            batch_size=B,
            d_latent=d_latent,
            max_seq_len=S_max
        )

        # 1. Prefill 4 tokens
        x_prefill = torch.randn(B, 4, d_model)
        out_prefill, c_prefill = mla_layer(x_prefill, mla_cache=mla_cache)
        self.assertEqual(out_prefill.shape, (B, 4, d_model))
        self.assertEqual(c_prefill.shape, (B, 4, d_latent))

        # 2. Step decoding at pos 4
        x_step = torch.randn(B, 1, d_model)
        out_step, c_step = mla_layer(x_step, mla_cache=mla_cache, pos=4)
        self.assertEqual(out_step.shape, (B, 1, d_model))
        self.assertFalse(torch.isnan(out_step).any())
        self.assertFalse(torch.isinf(out_step).any())


if __name__ == "__main__":
    unittest.main()

