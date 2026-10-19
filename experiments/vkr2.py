#!/usr/bin/env python3
"""
vkr2.py -- KV repair for depth-adaptive Transformers, v2 (single file).

Supported architectures: Qwen2 family, Llama family (including Llama-3.2-1B).

Changes vs vkr.py
  * Reuses existing adapters (adept_hsm_L*.pt, vkr_L*.pt) from --ckpt_dir; trains only what is missing.
  * NEW method `kvc`: per-layer corrector for BOTH Keys and Values (parameter-matched to vkr).
  * Fixed held-out domain: C4 via streaming `allenai/c4` (fallback: FineWeb-Edu stream).
  * Bootstrap 95% CIs over eval sequences, plus PAIRED CIs for (method - adept_hsm).
  * Dropped the misleading "gap closed %"; reports KL reduction vs native and the Value share of
    the Key/Value error instead. oracleK / oracleV are diagnostics, NOT ceilings.

Typical Kaggle commands (adapters from the first run are picked up automatically):
  # Qwen2.5-3B (36 layers): exit at layers 18, 24, 30
  python experiments/vkr2.py --model Qwen/Qwen2.5-3B --depths 18,24,30 --eval_n 120 \
      --ckpt_dir /kaggle/working/kv-cache-continuity/vkr_out

  # Llama-3.2-1B (16 layers): exit at layers 8, 10, 12
  python experiments/vkr2.py --model meta-llama/Llama-3.2-1B --depths 8,10,12 --eval_n 120 \
      --ckpt_dir /kaggle/working/kv-cache-continuity/vkr_out

Other modes
  --eval_only      never train; fail if an adapter checkpoint is missing
  --retrain        ignore existing checkpoints
  --rollout_n 8    also run free-running greedy rollouts (slow)
  --selftest       tiny random model on CPU, no downloads
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--depths", default="18,24,30",
                   help="comma list; ints = exit layer, floats with '.' = fraction of depth")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--train_steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--rank", type=int, default=64, help="VKR rank (hsm/kvc are parameter-matched to it)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5, help="weight of normalised V-MSE aux loss")
    p.add_argument("--eval_n", type=int, default=120, help="eval sequences per domain")
    p.add_argument("--eval_bs", type=int, default=6)
    p.add_argument("--rollout_n", type=int, default=0)
    p.add_argument("--rollout_new", type=int, default=64)
    p.add_argument("--rollout_prompt", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="vkr_out")
    p.add_argument("--ckpt_dir", default=None, help="where adapters live/are saved (default: --out)")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--no_c4", action="store_true")
    p.add_argument("--c4_docs", type=int, default=800)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- model wrapper
class Ctx:
    def __init__(self, model):
        self.m = model
        cfg = model.config
        self.layers = model.model.layers
        self.N = len(self.layers)
        self.H = cfg.num_attention_heads
        self.Hkv = cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.d = cfg.hidden_size
        self.rep = self.H // self.Hkv
        self.kvdim = self.Hkv * self.D
        # rotary_emb lives on model.model in Qwen2/recent Llama, or on each
        # self_attn layer in older transformers versions of Llama
        self.rotary = getattr(model.model, "rotary_emb", None)
        if self.rotary is None:
            # Fallback: try to grab it from the first layer's self_attn
            self.rotary = getattr(self.layers[0].self_attn, "rotary_emb", None)
        if self.rotary is None:
            raise RuntimeError(
                "rotary_emb not found on model.model or layers[0].self_attn; "
                "need transformers >= 4.48")
        self.dtype = next(model.parameters()).dtype

    def embed(self, ids):
        B, T = ids.shape
        h = self.m.model.embed_tokens(ids)
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary(h, pos)
        return h, cos, sin

    @staticmethod
    def rope(x, cos, sin):
        D = x.shape[-1]
        x1, x2 = x[..., : D // 2], x[..., D // 2:]
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos.unsqueeze(1) + rot * sin.unsqueeze(1)

    def split(self, x, nh):
        B, T, _ = x.shape
        return x.view(B, T, nh, self.D).transpose(1, 2)

    def rep_kv(self, x):
        return x if self.rep == 1 else x.repeat_interleave(self.rep, dim=1)

    def project_kv(self, i, h_src, cos, sin, dk=None, dv=None):
        """Native projection through layer i's own norm + W_K/W_V (+RoPE on K).
        dk / dv: optional additive corrections (B,T,kvdim) applied BEFORE head split / RoPE."""
        lay = self.layers[i]
        a = lay.self_attn
        n = lay.input_layernorm(h_src)
        kl, vl = a.k_proj(n), a.v_proj(n)
        if dk is not None:
            kl = kl + dk.to(kl.dtype)
        if dv is not None:
            vl = vl + dv.to(vl.dtype)
        return self.rope(self.split(kl, self.Hkv), cos, sin), self.split(vl, self.Hkv)

    def layer_fwd(self, i, h, cos, sin, kv_hat=None):
        lay = self.layers[i]
        a = lay.self_attn
        B, T, _ = h.shape
        n = lay.input_layernorm(h)
        q = self.rope(self.split(a.q_proj(n), self.H), cos, sin)
        k = self.rope(self.split(a.k_proj(n), self.Hkv), cos, sin)
        v = self.split(a.v_proj(n), self.Hkv)
        kh, vh = (k, v) if kv_hat is None else kv_hat
        khr, vhr = self.rep_kv(kh), self.rep_kv(vh)
        kr, vr = self.rep_kv(k), self.rep_kv(v)
        scale = self.D ** -0.5
        sc = torch.matmul(q, khr.transpose(-1, -2)).float() * scale
        diag = (q.float() * kr.float()).sum(-1) * scale
        eye = torch.eye(T, dtype=torch.bool, device=h.device)
        sc = torch.where(eye, diag.unsqueeze(-1), sc)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
        sc = sc.masked_fill(causal, float("-inf"))
        A = torch.softmax(sc, dim=-1)
        a_diag = torch.diagonal(A, dim1=-2, dim2=-1)
        out = torch.matmul(A.to(vhr.dtype), vhr) + a_diag.unsqueeze(-1).to(vhr.dtype) * (vr - vhr)
        out = out.transpose(1, 2).reshape(B, T, self.H * self.D)
        h = h + a.o_proj(out)
        h = h + lay.mlp(lay.post_attention_layernorm(h))
        return h, (k, v)

    def head(self, h, last_only=False):
        if last_only:
            h = h[:, -1:]
        return self.m.lm_head(self.m.model.norm(h))

    @torch.no_grad()
    def oracle(self, ids):
        h, cos, sin = self.embed(ids)
        hs, kv = [h], []
        for i in range(self.N):
            h, kvi = self.layer_fwd(i, h, cos, sin)
            hs.append(h)
            kv.append(kvi)
        return dict(hs=hs, kv=kv, logits=self.head(h), cos=cos, sin=sin)

    def approx(self, orc, L, kv_hats, exact_mask=None, last_only=False):
        h, cos, sin = orc["hs"][L], orc["cos"], orc["sin"]
        for j, i in enumerate(range(L, self.N)):
            kh, vh = kv_hats[j]
            if exact_mask is not None:
                m = exact_mask.view(1, 1, -1, 1)
                kh = torch.where(m, orc["kv"][i][0], kh)
                vh = torch.where(m, orc["kv"][i][1], vh)
            h, _ = self.layer_fwd(i, h, cos, sin, (kh, vh))
        return self.head(h, last_only)


# ----------------------------------------------------------------------------- KV methods
class Native:
    name = "native"

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        return [ctx.project_kv(i, h, orc["cos"], orc["sin"]) for i in range(L, ctx.N)]


class Direct:
    name = "direct"

    def kv(self, ctx, orc, L):
        k, v = orc["kv"][L - 1]
        return [(k, v) for _ in range(L, ctx.N)]


class OracleV:  # diagnostic: exact Values, projected Keys  (residual error = KEY error)
    name = "oracleV"

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        return [(ctx.project_kv(i, h, orc["cos"], orc["sin"])[0], orc["kv"][i][1]) for i in range(L, ctx.N)]


class OracleK:  # diagnostic: exact Keys, projected Values (residual error = VALUE error)
    name = "oracleK"

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        return [(orc["kv"][i][0], ctx.project_kv(i, h, orc["cos"], orc["sin"])[1]) for i in range(L, ctx.N)]


class HSM(nn.Module):
    """ADEPT-style: one shared mapper h_L -> h_hat, then each skipped layer's own W_K/W_V."""
    name = "adept_hsm"

    def __init__(self, d, width):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, width)
        self.w2 = nn.Linear(width, d)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        h32 = h.float()
        hh = (h32 + self.w2(F.gelu(self.w1(self.ln(h32))))).to(h.dtype)
        return [ctx.project_kv(i, hh, orc["cos"], orc["sin"]) for i in range(L, ctx.N)]


class VKR(nn.Module):
    """native Keys + learned low-rank Value correction."""
    name = "vkr"

    def __init__(self, d, n_skipped, rank, kvdim):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, rank)
        self.emb = nn.Embedding(n_skipped, rank)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.up = nn.Parameter(torch.zeros(n_skipped, rank, kvdim))

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        B, T, _ = h.shape
        z = self.down(self.ln(h.float()))
        z = F.gelu(z.unsqueeze(2) + self.emb.weight.unsqueeze(0).unsqueeze(0))
        delta = torch.einsum("btsr,srk->btsk", z, self.up)
        out = []
        for s, i in enumerate(range(L, ctx.N)):
            k, v = ctx.project_kv(i, h, orc["cos"], orc["sin"])
            dv = delta[:, :, s].to(v.dtype).reshape(B, T, ctx.Hkv, ctx.D).transpose(1, 2)
            out.append((k, v + dv))
        return out


class KVC(nn.Module):
    """NEW: learned low-rank corrections for BOTH Keys (pre-RoPE) and Values, per skipped layer."""
    name = "kvc"

    def __init__(self, d, n_skipped, rank, kvdim):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, rank)
        self.emb = nn.Embedding(n_skipped, rank)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.up_k = nn.Parameter(torch.zeros(n_skipped, rank, kvdim))
        self.up_v = nn.Parameter(torch.zeros(n_skipped, rank, kvdim))

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        z = self.down(self.ln(h.float()))
        z = F.gelu(z.unsqueeze(2) + self.emb.weight.unsqueeze(0).unsqueeze(0))
        dK = torch.einsum("btsr,srk->btsk", z, self.up_k)
        dV = torch.einsum("btsr,srk->btsk", z, self.up_v)
        return [ctx.project_kv(i, h, orc["cos"], orc["sin"], dk=dK[:, :, s], dv=dV[:, :, s])
                for s, i in enumerate(range(L, ctx.N))]


def n_params(m):
    return sum(p.numel() for p in m.parameters()) if isinstance(m, nn.Module) else 0


def kvc_rank(target, d, S, kvdim):
    # params = 2d + d*r + r + S*r + 2*S*r*kvdim
    return max(4, int(round((target - 2 * d) / (d + 1 + S + 2 * S * kvdim))))


# ----------------------------------------------------------------------------- losses / metrics
def kl_tokens(ref_logits, logits):
    lp_ref = F.log_softmax(ref_logits.float(), dim=-1)
    lp = F.log_softmax(logits.float(), dim=-1)
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1)


def vmse(orc, kvh, L):
    num = den = 0.0
    for j, i in enumerate(range(L, len(orc["kv"]))):
        v_star = orc["kv"][i][1].float()
        num = num + ((kvh[j][1].float() - v_star) ** 2).mean()
        den = den + (v_star ** 2).mean()
    return num / den


def make_scaler(enabled, device):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_method(ctx, method, L, data, args, device):
    params = [p for p in method.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.train_steps
    warm = max(1, min(50, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    scaler = make_scaler(ctx.dtype == torch.float16 and device.type == "cuda", device)
    t0 = time.time()
    every = max(1, total // 10)
    run_kl = run_v = 0.0
    for step in range(1, total + 1):
        ids = data.sample(args.batch).to(device)
        orc = ctx.oracle(ids)
        kvh = method.kv(ctx, orc, L)
        logits = ctx.approx(orc, L, kvh)
        kl = kl_tokens(orc["logits"][:, :-1], logits[:, :-1]).mean()
        vm = vmse(orc, kvh, L)
        loss = kl + args.alpha * vm
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_kl += kl.item()
        run_v += vm.item()
        if step % every == 0 or step == total:
            n = every if step % every == 0 else step % every
            print(f"    [{method.name} L={L}] step {step}/{total}  KL {run_kl / n:.4f}  Vmse {run_v / n:.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
            run_kl = run_v = 0.0


@torch.no_grad()
def evaluate(ctx, method, L, chunks, bs, device):
    kl_seq, top1_seq = [], []
    nll = onll = 0.0
    cnt = 0
    for s in range(0, len(chunks), bs):
        ids = chunks[s:s + bs].to(device)
        orc = ctx.oracle(ids)
        lg = orc["logits"] if method is None else ctx.approx(orc, L, method.kv(ctx, orc, L))
        lg, rl, tgt = lg[:, :-1], orc["logits"][:, :-1], ids[:, 1:]
        kl_seq += kl_tokens(rl, lg).mean(1).cpu().tolist()
        top1_seq += (lg.argmax(-1) == rl.argmax(-1)).float().mean(1).cpu().tolist()
        V = lg.shape[-1]
        nll += F.cross_entropy(lg.float().reshape(-1, V), tgt.reshape(-1), reduction="sum").item()
        onll += F.cross_entropy(rl.float().reshape(-1, V), tgt.reshape(-1), reduction="sum").item()
        cnt += tgt.numel()
    ppl, oppl = math.exp(nll / cnt), math.exp(onll / cnt)
    kl_seq, top1_seq = np.array(kl_seq), np.array(top1_seq)
    return dict(kl=float(kl_seq.mean()), top1=100 * float(top1_seq.mean()), ppl=ppl, oracle_ppl=oppl,
                dppl=ppl - oppl, kl_seq=kl_seq)


def boot_ci(x, rng, n=2000):
    x = np.asarray(x)
    m = x[rng.randint(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ----------------------------------------------------------------------------- data
def stream_texts(repo, n_docs, **kw):
    from datasets import load_dataset
    ds = load_dataset(repo, streaming=True, **kw)
    texts = []
    for r in ds:
        t = r.get("text")
        if t and len(t) > 200:
            texts.append(t)
        if len(texts) >= n_docs:
            break
    if len(texts) < 20:
        raise RuntimeError("too few documents")
    return texts


class Data:
    def __init__(self, tok, args, vocab, need_train):
        T, rng = args.seq_len, random.Random(args.seed)
        self.T = T
        self.eval = {}
        self.train = None
        if args.selftest:
            g = torch.Generator().manual_seed(args.seed)
            self.train = torch.randint(0, vocab, (T * 400,), generator=g)
            for dom in ("wikitext", "c4"):
                self.eval[dom] = torch.randint(0, vocab, (args.eval_n, T), generator=g)
            return
        from datasets import load_dataset
        enc = lambda txt: torch.tensor(tok(txt, add_special_tokens=False)["input_ids"])
        try:
            va = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        except Exception:
            va = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        if need_train:
            try:
                tr = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            except Exception:
                tr = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            self.train = enc("\n\n".join(x for x in tr["text"] if x.strip()))
        self.eval["wikitext"] = self._chunks(enc("\n\n".join(x for x in va["text"] if x.strip())), T, args.eval_n, rng)
        if not args.no_c4:
            eos = tok.eos_token_id if tok.eos_token_id is not None else 0
            for label, repo, kw in (("c4", "allenai/c4", dict(name="en", split="validation")),
                                    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", dict(name="sample-10BT", split="train"))):
                try:
                    texts = stream_texts(repo, args.c4_docs, **kw)
                    ids = []
                    for t in texts:
                        ids += tok(t, add_special_tokens=False)["input_ids"] + [eos]
                    self.eval[label] = self._chunks(torch.tensor(ids), T, args.eval_n, rng)
                    print(f"  held-out web domain loaded: {label} ({repo}), {len(self.eval[label])} sequences")
                    break
                except Exception as e:  # noqa
                    print(f"  ! {repo} unavailable ({type(e).__name__}: {str(e)[:120]})")
            else:
                print("  ! NO held-out web domain could be loaded; only WikiText will be evaluated")
        print(f"  eval domains: {list(self.eval)}" + (f"; train tokens: {len(self.train):,}" if self.train is not None else ""))

    @staticmethod
    def _chunks(ids, T, n, rng):
        k = len(ids) // T
        idx = list(range(k))
        rng.shuffle(idx)
        idx = idx[: min(n, k)]
        return torch.stack([ids[i * T:(i + 1) * T] for i in idx])

    def sample(self, B):
        st = torch.randint(0, len(self.train) - self.T - 1, (B,))
        return torch.stack([self.train[s:s + self.T] for s in st])


# ----------------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout(ctx, method, L, prompts, n_new, device):
    outs = []
    for p in prompts:
        cur = p.unsqueeze(0).to(device)
        plen = cur.shape[1]
        for _ in range(n_new):
            orc = ctx.oracle(cur)
            if method is None:
                lg = orc["logits"][:, -1]
            else:
                mask = torch.arange(cur.shape[1], device=device) < plen
                lg = ctx.approx(orc, L, method.kv(ctx, orc, L), exact_mask=mask, last_only=True)[:, -1]
            cur = torch.cat([cur, lg.argmax(-1, keepdim=True)], dim=1)
        outs.append(cur[0, plen:].cpu())
    return outs


def rollout_metrics(ref, gen, k):
    m, fd = [], []
    for r, g in zip(ref, gen):
        eq = (r == g)
        m.append(100 * eq[:k].float().mean().item())
        bad = (~eq).nonzero()
        fd.append(bad[0].item() if len(bad) else len(r))
    return dict(match=sum(m) / len(m), first_div=sum(fd) / len(fd))


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    set_seed(args.seed)
    # Default output dir: "llamavkrout" for Llama models, "vkr_out" for others
    if args.out == "vkr_out" and "llama" in args.model.lower():
        args.out = "llamavkrout"
    ckpt_dir = args.ckpt_dir or args.out
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.selftest else "cpu")

    if args.selftest:
        # Support both Qwen2 and Llama selftest variants
        is_llama_selftest = "llama" in args.model.lower()
        if is_llama_selftest:
            from transformers import LlamaConfig, LlamaForCausalLM
            cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=8,
                              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                              rope_theta=10000.0, tie_word_embeddings=False)
            model = LlamaForCausalLM(cfg).float().eval()
        else:
            from transformers import Qwen2Config, Qwen2ForCausalLM
            cfg = Qwen2Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=8,
                              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                              rope_theta=10000.0, tie_word_embeddings=False)
            model = Qwen2ForCausalLM(cfg).float().eval()
        tok = None
        if args.depths == "18,24,30":
            args.depths = "0.5,0.75"
        args.train_steps = min(args.train_steps, 40)
        args.eval_n, args.eval_bs, args.seq_len = 16, 4, 48
        args.rollout_prompt, args.rollout_new = 8, 8
        args.rollout_n = args.rollout_n or 2
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        if device.type == "cuda":
            dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dt = torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt)
        model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    ctx = Ctx(model)
    selftest_label = ('tiny-random-llama' if 'llama' in args.model.lower() else 'tiny-random-qwen2') if args.selftest else None
    print(f"model={selftest_label or args.model} layers={ctx.N} d={ctx.d} "
          f"H={ctx.H} Hkv={ctx.Hkv} D={ctx.D} dtype={ctx.dtype} device={device}")
    depths = []
    for x in args.depths.split(","):
        x = x.strip()
        L = int(round(float(x) * ctx.N)) if "." in x else int(x)
        assert 1 <= L < ctx.N, f"bad depth {L}"
        depths.append(L)

    data = Data(tok, args, model.config.vocab_size, need_train=not args.eval_only)

    # ---- sanity checks
    first_dom = list(data.eval)[0]
    ids = data.eval[first_dom][:2].to(device)
    orc = ctx.oracle(ids)
    ref = model(ids).logits
    k0 = kl_tokens(ref[:, :-1], orc["logits"][:, :-1]).mean().item()
    Ls = depths[0]
    lg = ctx.approx(orc, Ls, Native().kv(ctx, orc, Ls), exact_mask=torch.ones(ids.shape[1], dtype=torch.bool, device=device))
    k1 = kl_tokens(orc["logits"][:, :-1], lg[:, :-1]).mean().item()
    print(f"sanity: KL(HF vs manual)={k0:.2e}  KL(all-exact approx vs oracle)={k1:.2e}")
    tol = 0.05 if ctx.dtype != torch.float32 else 1e-4
    if k0 > tol or k1 > tol:
        raise SystemExit("SANITY CHECK FAILED: manual forward does not reproduce the HF model. Aborting.")

    rng = np.random.RandomState(0)
    results, rollouts = {}, {}
    for L in depths:
        print(f"\n=== exit depth L={L}/{ctx.N} ({100 * L / ctx.N:.0f}%) ===")
        S = ctx.N - L
        vkr = VKR(ctx.d, S, args.rank, ctx.kvdim).to(device)
        width = max(8, int(round(n_params(vkr) / (2 * ctx.d + 1))))
        hsm = HSM(ctx.d, width).to(device)
        kvc = KVC(ctx.d, S, kvc_rank(n_params(vkr), ctx.d, S, ctx.kvdim), ctx.kvdim).to(device)
        print(f"  params: VKR={n_params(vkr):,}  ADEPT-HSM={n_params(hsm):,}  KVC={n_params(kvc):,} "
              f"(rank {kvc.down.out_features})")
        for m in (hsm, vkr, kvc):
            path = os.path.join(ckpt_dir, f"{m.name}_L{L}.pt")
            if os.path.exists(path) and not args.retrain:
                m.load_state_dict(torch.load(path, map_location=device))
                print(f"  loaded {path}")
            elif args.eval_only:
                raise SystemExit(f"--eval_only but checkpoint missing: {path}")
            else:
                print(f"  training {m.name} (no checkpoint at {path})")
                train_method(ctx, m, L, data, args, device)
                torch.save(m.state_dict(), path)
            m.eval()
        methods = {"oracle": None, "direct": Direct(), "native": Native(), "oracleK": OracleK(),
                   "oracleV": OracleV(), "adept_hsm": hsm, "vkr": vkr, "kvc": kvc}
        results[L] = {}
        for dom, chunks in data.eval.items():
            res = {n: evaluate(ctx, m, L, chunks, args.eval_bs, device) for n, m in methods.items()}
            base = res["adept_hsm"]["kl_seq"]
            nat = res["native"]["kl"]
            for n, r in res.items():
                mean, lo, hi = boot_ci(r["kl_seq"], rng)
                r["kl_ci"] = (lo, hi)
                r["params"] = n_params(methods[n]) if methods[n] is not None else 0
                r["kl_red_vs_native"] = 100 * (nat - r["kl"]) / nat if nat > 1e-12 else float("nan")
                if n in ("vkr", "kvc"):
                    d_mean, d_lo, d_hi = boot_ci(r["kl_seq"] - base, rng)
                    r["dkl_vs_hsm"] = (d_mean, d_lo, d_hi)
            ek, ev = res["oracleK"]["kl"], res["oracleV"]["kl"]
            v_share = 100 * ek / (ek + ev) if (ek + ev) > 0 else float("nan")
            results[L][dom] = dict(methods={n: {k: v for k, v in r.items() if k != "kl_seq"} for n, r in res.items()},
                                   value_share_of_error=v_share, n_seq=len(chunks))
            print(f"  [{dom}]  N={len(chunks)} seqs x {args.seq_len} tokens   (Value share of Key+Value error: {v_share:.0f}%)")
            print(f"    {'method':10s} {'KL [95% CI]':>24s} {'top1%':>7s} {'PPL':>8s} {'dPPL':>7s} {'KLred%':>7s}  {'dKL vs hsm [95% CI]':>26s} {'params':>9s}")
            for n, r in res.items():
                dk = r.get("dkl_vs_hsm")
                dks = f"{dk[0]:+.4f} [{dk[1]:+.4f},{dk[2]:+.4f}]" if dk else ""
                print(f"    {n:10s} {r['kl']:8.4f} [{r['kl_ci'][0]:.4f},{r['kl_ci'][1]:.4f}] {r['top1']:7.2f} "
                      f"{r['ppl']:8.2f} {r['dppl']:7.2f} {r['kl_red_vs_native']:7.1f}  {dks:>26s} {r['params']:9,d}")

        if args.rollout_n > 0:
            dom0 = list(data.eval)[0]
            prompts = [c[: args.rollout_prompt] for c in data.eval[dom0][: args.rollout_n]]
            refgen = rollout(ctx, None, L, prompts, args.rollout_new, device)
            rollouts[L] = {}
            print(f"  free-running rollout on {dom0} ({len(prompts)} prompts, {args.rollout_new} new tokens)")
            for n in ("direct", "native", "adept_hsm", "vkr", "kvc"):
                gen = rollout(ctx, methods[n], L, prompts, args.rollout_new, device)
                rm = rollout_metrics(refgen, gen, args.rollout_new)
                rollouts[L][n] = rm
                print(f"    {n:10s} Match@{args.rollout_new} {rm['match']:6.2f}%   first-divergence step {rm['first_div']:.1f}")

    with open(os.path.join(args.out, "results_v2.json"), "w") as f:
        json.dump(dict(args=vars(args), results=results, rollouts=rollouts), f, indent=2)
    lines = ["# KV repair results (v2)", "", f"model `{args.model}`, steps={args.train_steps}, rank={args.rank}", ""]
    for L, doms in results.items():
        for dom, d in doms.items():
            lines += [f"## L={L} ({dom}), N={d['n_seq']}, Value share of error {d['value_share_of_error']:.0f}%", "",
                      "| method | KL [95% CI] | top1 % | PPL | dPPL | KL red. vs native % | dKL vs hsm [95% CI] | params |",
                      "|---|---|---|---|---|---|---|---|"]
            for n, r in d["methods"].items():
                dk = r.get("dkl_vs_hsm")
                dks = f"{dk[0]:+.4f} [{dk[1]:+.4f}, {dk[2]:+.4f}]" if dk else ""
                lines.append(f"| {n} | {r['kl']:.4f} [{r['kl_ci'][0]:.4f}, {r['kl_ci'][1]:.4f}] | {r['top1']:.2f} | "
                             f"{r['ppl']:.2f} | {r['dppl']:.2f} | {r['kl_red_vs_native']:.1f} | {dks} | {r['params']:,} |")
            lines.append("")
    open(os.path.join(args.out, "results_v2.md"), "w").write("\n".join(lines))
    print(f"\nDone. Wrote {args.out}/results_v2.json and results_v2.md (adapters in {ckpt_dir}).")
    print("dKL vs hsm < 0 with a CI that excludes 0 means that method is significantly better than the ADEPT-style mapper.")


if __name__ == "__main__":
    main()