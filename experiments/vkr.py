#!/usr/bin/env python3
"""
vkr.py -- Value-targeted KV Repair for depth-adaptive Transformers (single file).

What it does (all in one run):
  1. Loads a frozen decoder-only LM (Qwen2 / Llama family).
  2. Implements the teacher-forced "attention-only" protocol from your paper:
     tokens exit at depth L, upper-layer KV slots of the HISTORY are approximated,
     the current token's own K/V stays exact, readout is full-depth.
  3. Compares, per exit depth L and per domain:
        oracle      full model (reference)
        direct      Direct-Copy floor (K_L, V_L repeated)
        native      Native Projection = CALM state propagation (your paper)
        oracleK     exact Keys,  projected Values   (diagnostic)
        oracleV     projected Keys, exact Values    (ceiling for any Value fix)
        adept_hsm   ADEPT-style baseline: learned hidden-state mapper h_L -> h_hat,
                    then every skipped layer's own W_K/W_V (parameter-matched)
        vkr         OURS: native Keys + learned low-rank Value correction
  4. Optionally runs free-running greedy rollouts (--rollout_n > 0).
  5. Writes results.json, results.md and adapter checkpoints.

Usage
  python vkr.py --selftest                       # CPU, tiny random model, no downloads
  python vkr.py --model Qwen/Qwen2.5-3B          # full run (Colab/Kaggle T4 OK)
  python vkr.py --model Qwen/Qwen2.5-0.5B --depths 0.5,0.67,0.83 --train_steps 800
  python vkr.py --model Qwen/Qwen2.5-3B --rollout_n 8

Requires: torch, transformers>=4.48, datasets (datasets not needed for --selftest).
"""
import argparse
import json
import math
import os
import random
import time

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
    p.add_argument("--rank", type=int, default=64, help="VKR bottleneck rank")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5, help="weight of normalised V-MSE aux loss")
    p.add_argument("--eval_n", type=int, default=60, help="eval sequences per domain")
    p.add_argument("--eval_bs", type=int, default=6)
    p.add_argument("--rollout_n", type=int, default=0, help="# prompts for free-running rollout (0=off)")
    p.add_argument("--rollout_new", type=int, default=64)
    p.add_argument("--rollout_prompt", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="vkr_out")
    p.add_argument("--no_c4", action="store_true", help="skip the held-out C4 domain")
    p.add_argument("--selftest", action="store_true", help="tiny random model on CPU, no downloads")
    return p.parse_args()


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- model wrapper
class Ctx:
    """Manual layer-by-layer forward so we can control the KV used for history."""

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
        self.rotary = getattr(model.model, "rotary_emb", None)
        if self.rotary is None:
            raise RuntimeError("model.model.rotary_emb not found; need transformers >= 4.48")
        self.dtype = next(model.parameters()).dtype

    # --- helpers
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

    def project_kv(self, i, h_src, cos, sin):
        """Native projection of a hidden state through layer i's own norm + W_K/W_V (+RoPE on K)."""
        lay = self.layers[i]
        a = lay.self_attn
        n = lay.input_layernorm(h_src)
        k = self.rope(self.split(a.k_proj(n), self.Hkv), cos, sin)
        v = self.split(a.v_proj(n), self.Hkv)
        return k, v

    # --- one decoder layer; history uses kv_hat, own position stays exact
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
        sc = torch.matmul(q, khr.transpose(-1, -2)).float() * scale            # (B,H,T,T)
        diag = (q.float() * kr.float()).sum(-1) * scale                        # exact self-score
        eye = torch.eye(T, dtype=torch.bool, device=h.device)
        sc = torch.where(eye, diag.unsqueeze(-1), sc)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
        sc = sc.masked_fill(causal, float("-inf"))
        A = torch.softmax(sc, dim=-1)
        a_diag = torch.diagonal(A, dim1=-2, dim2=-1)                           # (B,H,T)
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
        """Run layers L..N-1 for the current positions, attending over approximated history KV."""
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


class OracleV:
    name = "oracleV"

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        return [(ctx.project_kv(i, h, orc["cos"], orc["sin"])[0], orc["kv"][i][1]) for i in range(L, ctx.N)]


class OracleK:
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
    """OURS: native Keys + learned low-rank Value correction (shared down-proj, per-layer up-proj)."""
    name = "vkr"

    def __init__(self, d, n_skipped, rank, kvdim):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, rank)
        self.emb = nn.Embedding(n_skipped, rank)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.up = nn.Parameter(torch.zeros(n_skipped, rank, kvdim))   # zero-init => starts == native

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        B, T, _ = h.shape
        z = self.down(self.ln(h.float()))                                   # (B,T,r)
        z = F.gelu(z.unsqueeze(2) + self.emb.weight.unsqueeze(0).unsqueeze(0))   # (B,T,S,r)
        delta = torch.einsum("btsr,srk->btsk", z, self.up)                  # (B,T,S,kvdim)
        out = []
        for s, i in enumerate(range(L, ctx.N)):
            k, v = ctx.project_kv(i, h, orc["cos"], orc["sin"])
            dv = delta[:, :, s].to(v.dtype).reshape(B, T, ctx.Hkv, ctx.D).transpose(1, 2)
            out.append((k, v + dv))
        return out


def n_params(m):
    return sum(p.numel() for p in m.parameters()) if isinstance(m, nn.Module) else 0


# ----------------------------------------------------------------------------- losses / metrics
def kl_tokens(ref_logits, logits):
    """forward KL(ref || approx) per token, float32."""
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


def train_method(ctx, method, L, data, args, device):
    params = [p for p in method.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.train_steps
    warm = max(1, min(50, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    use_scaler = ctx.dtype == torch.float16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    t0 = time.time()
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
        if step % max(1, total // 10) == 0 or step == total:
            n = max(1, total // 10) if step % max(1, total // 10) == 0 else step % max(1, total // 10)
            print(f"    [{method.name} L={L}] step {step}/{total}  KL {run_kl / n:.4f}  Vmse {run_v / n:.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
            run_kl = run_v = 0.0


@torch.no_grad()
def evaluate(ctx, method, L, chunks, bs, device):
    kl_sum = top1 = nll = onll = 0.0
    cnt = 0
    for s in range(0, len(chunks), bs):
        ids = chunks[s:s + bs].to(device)
        orc = ctx.oracle(ids)
        if method is None:
            lg = orc["logits"]
        else:
            lg = ctx.approx(orc, L, method.kv(ctx, orc, L))
        lg, rl, tgt = lg[:, :-1], orc["logits"][:, :-1], ids[:, 1:]
        kl_sum += kl_tokens(rl, lg).sum().item()
        top1 += (lg.argmax(-1) == rl.argmax(-1)).sum().item()
        V = lg.shape[-1]
        nll += F.cross_entropy(lg.float().reshape(-1, V), tgt.reshape(-1), reduction="sum").item()
        onll += F.cross_entropy(rl.float().reshape(-1, V), tgt.reshape(-1), reduction="sum").item()
        cnt += tgt.numel()
    ppl, oppl = math.exp(nll / cnt), math.exp(onll / cnt)
    return dict(kl=kl_sum / cnt, top1=100 * top1 / cnt, ppl=ppl, oracle_ppl=oppl, dppl=ppl - oppl)


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, tok, args, vocab):
        T, rng = args.seq_len, random.Random(args.seed)
        self.T = T
        self.eval = {}
        if args.selftest:
            g = torch.Generator().manual_seed(args.seed)
            self.train = torch.randint(0, vocab, (T * 400,), generator=g)
            for dom in ("wikitext", "c4"):
                self.eval[dom] = torch.randint(0, vocab, (args.eval_n, T), generator=g)
            return
        from datasets import load_dataset
        try:
            tr = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            va = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        except Exception:
            tr = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            va = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        enc = lambda txt: torch.tensor(tok(txt, add_special_tokens=False)["input_ids"])
        self.train = enc("\n\n".join(x for x in tr["text"] if x.strip()))
        self.eval["wikitext"] = self._chunks(enc("\n\n".join(x for x in va["text"] if x.strip())), T, args.eval_n, rng)
        if not args.no_c4:
            try:
                c4 = load_dataset("stas/c4-en-10k", split="train")
                eos = tok.eos_token_id if tok.eos_token_id is not None else 0
                ids = []
                for r in c4.select(range(min(400, len(c4)))):
                    ids += tok(r["text"], add_special_tokens=False)["input_ids"] + [eos]
                self.eval["c4"] = self._chunks(torch.tensor(ids), T, args.eval_n, rng)
            except Exception as e:  # noqa
                print(f"  ! C4 unavailable ({e}); continuing without it")
        print(f"  train tokens: {len(self.train):,}; eval domains: {list(self.eval)}")

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
    """Free-running greedy generation. Prompt KV exact; generated tokens' upper-layer KV approximated.
    method=None => oracle (full-depth)."""
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
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.selftest else "cpu")

    if args.selftest:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=8,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                          rope_theta=10000.0, tie_word_embeddings=False)
        model = Qwen2ForCausalLM(cfg).float().eval()
        tok = None
        if args.depths == "18,24,30":
            args.depths = "0.5,0.75"
        args.train_steps = min(args.train_steps, 60)
        args.eval_n, args.eval_bs, args.seq_len = 8, 4, 48
        args.rollout_prompt, args.rollout_new = 8, 8
        args.rollout_n = args.rollout_n or 2
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        if device.type == "cuda":
            dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dt = torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    ctx = Ctx(model)
    print(f"model={args.model if not args.selftest else 'tiny-random-qwen2'} layers={ctx.N} d={ctx.d} "
          f"H={ctx.H} Hkv={ctx.Hkv} D={ctx.D} dtype={ctx.dtype} device={device}")
    depths = []
    for x in args.depths.split(","):
        x = x.strip()
        L = int(round(float(x) * ctx.N)) if "." in x else int(x)
        assert 1 <= L < ctx.N, f"bad depth {L}"
        depths.append(L)

    data = Data(tok, args, model.config.vocab_size)

    # ---- sanity checks (fail loudly if the manual forward is wrong)
    ids = data.eval["wikitext"][:2].to(device)
    orc = ctx.oracle(ids)
    ref = model(ids).logits
    k0 = kl_tokens(ref[:, :-1], orc["logits"][:, :-1]).mean().item()
    Ls = depths[0]
    lg = ctx.approx(orc, Ls, Native().kv(ctx, orc, Ls), exact_mask=torch.ones(ids.shape[1], dtype=torch.bool, device=device))
    k1 = kl_tokens(orc["logits"][:, :-1], lg[:, :-1]).mean().item()
    kn = kl_tokens(orc["logits"][:, :-1], ctx.approx(orc, Ls, Native().kv(ctx, orc, Ls))[:, :-1]).mean().item()
    print(f"sanity: KL(HF vs manual)={k0:.2e}  KL(all-exact approx vs oracle)={k1:.2e}  KL(native @L={Ls})={kn:.3f}")
    tol = 0.05 if ctx.dtype != torch.float32 else 1e-4
    if k0 > tol or k1 > tol:
        raise SystemExit("SANITY CHECK FAILED: manual forward does not reproduce the HF model. Aborting.")

    results, rollouts = {}, {}
    for L in depths:
        print(f"\n=== exit depth L={L}/{ctx.N} ({100 * L / ctx.N:.0f}%) ===")
        S = ctx.N - L
        vkr = VKR(ctx.d, S, args.rank, ctx.kvdim).to(device)
        width = max(8, int(round(n_params(vkr) / (2 * ctx.d + 1))))
        hsm = HSM(ctx.d, width).to(device)
        print(f"  params: VKR={n_params(vkr):,}  ADEPT-HSM={n_params(hsm):,} (width {width}, matched)")
        for m in (hsm, vkr):
            train_method(ctx, m, L, data, args, device)
            torch.save(m.state_dict(), os.path.join(args.out, f"{m.name}_L{L}.pt"))
        methods = {"oracle": None, "direct": Direct(), "native": Native(), "oracleK": OracleK(),
                   "oracleV": OracleV(), "adept_hsm": hsm, "vkr": vkr}
        results[L] = {}
        for dom, chunks in data.eval.items():
            res = {n: evaluate(ctx, m, L, chunks, args.eval_bs, device) for n, m in methods.items()}
            gap = res["native"]["kl"] - res["oracleV"]["kl"]
            for n in res:
                res[n]["params"] = n_params(methods[n]) if methods[n] is not None else 0
                res[n]["gap_closed"] = (100 * (res["native"]["kl"] - res[n]["kl"]) / gap) if gap > 1e-9 else float("nan")
            results[L][dom] = res
            print(f"  [{dom}]  (N={len(chunks)} seqs x {args.seq_len} tokens)")
            print(f"    {'method':10s} {'KL':>8s} {'top1%':>7s} {'PPL':>9s} {'dPPL':>8s} {'gap%':>7s} {'params':>10s}")
            for n, r in res.items():
                print(f"    {n:10s} {r['kl']:8.4f} {r['top1']:7.2f} {r['ppl']:9.2f} {r['dppl']:8.2f} "
                      f"{r['gap_closed']:7.1f} {r['params']:10,d}")

        if args.rollout_n > 0:
            dom0 = list(data.eval)[0]
            prompts = [c[: args.rollout_prompt] for c in data.eval[dom0][: args.rollout_n]]
            ref = rollout(ctx, None, L, prompts, args.rollout_new, device)
            rollouts[L] = {}
            print(f"  free-running rollout ({len(prompts)} prompts, {args.rollout_new} new tokens)")
            for n in ("direct", "native", "adept_hsm", "vkr"):
                gen = rollout(ctx, methods[n], L, prompts, args.rollout_new, device)
                rm = rollout_metrics(ref, gen, args.rollout_new)
                rollouts[L][n] = rm
                print(f"    {n:10s} Match@{args.rollout_new} {rm['match']:6.2f}%   first-divergence step {rm['first_div']:.1f}")

    # ---- write outputs
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(args=vars(args), results=results, rollouts=rollouts), f, indent=2)
    lines = ["# VKR results", "", f"model: `{args.model}`  rank={args.rank}  steps={args.train_steps}", ""]
    for L, doms in results.items():
        for dom, res in doms.items():
            lines += [f"## L={L}  ({dom})", "", "| method | KL | top1 % | PPL | dPPL | gap closed % | params |",
                      "|---|---|---|---|---|---|---|"]
            for n, r in res.items():
                lines.append(f"| {n} | {r['kl']:.4f} | {r['top1']:.2f} | {r['ppl']:.2f} | {r['dppl']:.2f} | "
                             f"{r['gap_closed']:.1f} | {r['params']:,} |")
            lines.append("")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(lines))
    print(f"\nDone. Wrote {args.out}/results.json, results.md and adapter checkpoints.")
    print("'gap closed %' = (KL_native - KL_method) / (KL_native - KL_oracleV): 100 means as good as exact Values.")


if __name__ == "__main__":
    main()