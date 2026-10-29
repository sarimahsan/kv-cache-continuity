"""ONE script that closes every open reviewer item.  Needs vkr2.py and vkr3.py beside it.

  --part fills    Do zero / mean / previous-token fills really differ from cache omission?  (item: "controls collapse")
  --part table1   Table I redone with N=200 (seq-level bootstrap CIs), PPL with robustness checks, per-token KL quantiles
  --part rollout  Table VI regenerated with an explicit Match@k / first-divergence definition + CIs (and Gen-NLL)
  --part latency  MEASURED per-token latency / memory of: full step, depth-L step, native fill, KVC, shadow, batched backfill
  --part all      everything (rollout is the slow one; use --rollout_n to shrink)

  python paper_fixes.py --model Qwen/Qwen2.5-3B --depths 18,24,30 --part all
  python paper_fixes.py --selftest            (tiny random model, CPU, checks the code paths)

Definitions it prints / uses (put these in the paper):
  Match@k   = % of the first k generated positions where the greedy token equals the full model's greedy token (POSITION-WISE,
              not prefix; after a divergence repeats can still match).      First divergence = index of the first mismatch.
  PPL       = exp(mean NLL of the ground-truth next token); robustness: median per-token NLL and the share of tokens with NLL>20.
  per-token KL quantiles are pooled over all N*(T-1) positions.
Writes paper_fixes_<model>.json
"""
import argparse, json, math, os, random, time, types
import numpy as np
import torch
import torch.nn.functional as F
import vkr2
from vkr2 import Ctx, Native, Direct, OracleK, OracleV, KVC, kl_tokens, boot_ci, set_seed, Data, rollout, rollout_metrics, gen_nll, kvc_rank, n_params

LN2 = math.log(2)


# ------------------------------------------------------------------------------------------------ context with protocol switches
class CtxF(Ctx):
    """self_exact=True : current token's own K/V in skipped layers is exact (repair-harness protocol)
       self_exact=False: it is replaced too (T=64 depth-sweep protocol)       omit=True: history in skipped layers is masked out"""
    self_exact, omit = True, False

    def layer_fwd(self, i, h, cos, sin, kv_hat=None):
        if kv_hat is None or (self.self_exact and not self.omit):
            return super().layer_fwd(i, h, cos, sin, kv_hat)
        lay, a = self.layers[i], self.layers[i].self_attn
        B, T, _ = h.shape
        n = lay.input_layernorm(h)
        q = self.rope(self.split(a.q_proj(n), self.H), cos, sin)
        k = self.rope(self.split(a.k_proj(n), self.Hkv), cos, sin)
        v = self.split(a.v_proj(n), self.Hkv)
        if self.omit:
            assert self.self_exact, "omission is defined with the current token exact (otherwise a token has no keys at all)"
            out = self.rep_kv(v)                                    # attention row = one-hot on itself
        else:
            khr, vhr = self.rep_kv(kv_hat[0]), self.rep_kv(kv_hat[1])
            sc = torch.matmul(q, khr.transpose(-1, -2)).float() * self.D ** -0.5
            sc = sc.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1), float("-inf"))
            out = torch.matmul(torch.softmax(sc, -1).to(vhr.dtype), vhr)
        out = out.transpose(1, 2).reshape(B, T, self.H * self.D)
        h = h + a.o_proj(out)
        h = h + lay.mlp(lay.post_attention_layernorm(h))
        return h, (k, v)


class Fill:                                   # zero / running-mean / previous-token fills  (all kept as tensors, so they are visibly distinct)
    def __init__(self, kind):
        self.kind, self.name = kind, kind

    def kv(self, ctx, orc, L):
        out = []
        for i in range(L, ctx.N):
            k, v = ctx.project_kv(i, orc["hs"][L], orc["cos"], orc["sin"])     # native, only used to build the fill
            if self.kind == "zero":
                out.append((torch.zeros_like(k), torch.zeros_like(v)))
            elif self.kind == "mean":                                            # causal running mean over EARLIER positions
                def rm(x):
                    c = x.float().cumsum(2)
                    c = torch.cat([torch.zeros_like(c[:, :, :1]), c[:, :, :-1]], 2)
                    cnt = torch.arange(x.shape[2], device=x.device).clamp(min=1).view(1, 1, -1, 1)
                    return (c / cnt).to(x.dtype)
                out.append((rm(k), rm(v)))
            elif self.kind == "prev":                                            # entry s := native K/V of token s-1
                sh = lambda x: torch.cat([torch.zeros_like(x[:, :, :1]), x[:, :, :-1]], 2)
                out.append((sh(k), sh(v)))
        return out


def get_data(tok, T, n, selftest, vocab):
    ns = types.SimpleNamespace(seq_len=T, eval_n=n, selftest=selftest, no_c4=True, c4_docs=0, seed=0)
    return Data(tok, ns, vocab, False).eval["wikitext"]


def ci(x, rng):
    m, lo, hi = boot_ci(x, rng)
    return [m, lo, hi]


# ------------------------------------------------------------------------------------------------ part: fills
@torch.no_grad()
def part_fills(ctx, tok, args, device, vocab):
    print("\n##### FILL CONTROLS (current token exact; WikiText-2, N=%d)" % args.n_fill)
    res, rng = {}, np.random.RandomState(0)
    ctx.omit = False
    for T in (64, 128):
        ch = get_data(tok, T if not args.selftest else 24 * T // 64, args.n_fill, args.selftest, vocab)
        for L in args.depths:
            methods = {"native": Native(), "direct": Direct(), "zero": Fill("zero"), "mean": Fill("mean"), "prev": Fill("prev"), "omit": Native()}
            kl = {k: [] for k in methods}
            top = {k: [] for k in methods}
            lgs = {}
            for s in range(0, len(ch), args.bs):
                ids = ch[s:s + args.bs].to(device)
                orc = ctx.oracle(ids)
                for name, m in methods.items():
                    ctx.omit = name == "omit"
                    lg = ctx.approx(orc, L, m.kv(ctx, orc, L))[:, :-1]
                    ctx.omit = False
                    rl = orc["logits"][:, :-1]
                    kl[name] += kl_tokens(rl, lg).mean(1).cpu().tolist()
                    top[name] += (lg.argmax(-1) == rl.argmax(-1)).float().mean(1).cpu().tolist()
                    if s == 0:
                        lgs[name] = lg.float().cpu()
            row = {k: dict(kl=ci(kl[k], rng), top1=100 * float(np.mean(top[k]))) for k in methods}
            row["max_abs_logit_diff_vs_omit"] = {k: float((lgs[k] - lgs["omit"]).abs().max()) for k in methods if k != "omit"}
            res[f"T{T}_L{L}"] = row
            print(f"\n T={T}  L={L}")
            print(f"   {'policy':8s} {'KL [95% CI]':26s} {'top1':>6s} {'max|dlogit| vs omit':>20s}")
            for k in methods:
                m_, lo, hi = row[k]["kl"]
                dd = row["max_abs_logit_diff_vs_omit"].get(k, 0.0)
                print(f"   {k:8s} {m_:7.3f} [{lo:6.3f},{hi:6.3f}]   {row[k]['top1']:6.1f} {dd:20.4f}")
    print("\n -> if two policies show identical KL AND max|dlogit|=0 the control is degenerate (state it); otherwise report them as distinct.")
    return res


# ------------------------------------------------------------------------------------------------ part: table1
@torch.no_grad()
def part_table1(ctx, tok, args, device, vocab):
    print("\n##### TABLE I REDONE  N=%d  T=%d  (self-approx = depth-sweep protocol; self-exact also shown)" % (args.n_t1, args.t1_len))
    ch = get_data(tok, args.t1_len if not args.selftest else 24, args.n_t1, args.selftest, vocab)
    rng = np.random.RandomState(0)
    res = {}
    for self_exact in (False, True):
        ctx.self_exact = self_exact
        for L in args.depths:
            acc = {k: dict(kl=[], top1=[], nll=[], tok_kl=[], tok_nll=[]) for k in ("oracle", "readout_only", "direct", "native")}
            for s in range(0, len(ch), args.bs):
                ids = ch[s:s + args.bs].to(device)
                orc = ctx.oracle(ids)
                tgt = ids[:, 1:]
                ref = orc["logits"][:, :-1]
                outs = {"oracle": ref, "readout_only": ctx.head(orc["hs"][L])[:, :-1],
                        "direct": ctx.approx(orc, L, Direct().kv(ctx, orc, L))[:, :-1],
                        "native": ctx.approx(orc, L, Native().kv(ctx, orc, L))[:, :-1]}
                for k, lg in outs.items():
                    kt = kl_tokens(ref, lg)
                    nll = F.cross_entropy(lg.float().transpose(1, 2), tgt, reduction="none")
                    acc[k]["kl"] += kt.mean(1).cpu().tolist()
                    acc[k]["top1"] += (lg.argmax(-1) == ref.argmax(-1)).float().mean(1).cpu().tolist()
                    acc[k]["nll"] += nll.mean(1).cpu().tolist()
                    acc[k]["tok_kl"].append(kt.flatten().cpu())
                    acc[k]["tok_nll"].append(nll.flatten().cpu())
            tag = "self_exact" if self_exact else "self_approx"
            print(f"\n [{tag}] L={L}/{ctx.N} ({100 * L / ctx.N:.0f}%)")
            print(f"   {'cond':13s} {'KL [95% CI]':24s} {'top1 [CI]':20s} {'PPL [CI]':24s} {'medNLL':>7s} {'%NLL>20':>8s} {'KL q50/q90/q99':>18s} {'%KL>ln2':>8s}")
            row = {}
            for k, a in acc.items():
                kl_, top_ = ci(a["kl"], rng), ci(100 * np.array(a["top1"]), rng)
                nl = np.array(a["nll"])
                ppl = [float(math.exp(min(x, 50))) for x in ci(nl, rng)]
                tn, tk = torch.cat(a["tok_nll"]).numpy(), torch.cat(a["tok_kl"]).numpy()
                q = np.percentile(tk, [50, 90, 99])
                row[k] = dict(kl=kl_, top1=top_, ppl=ppl, median_nll=float(np.median(tn)), frac_nll_gt20=float((tn > 20).mean()),
                              kl_q50_90_99=q.tolist(), frac_kl_gt_ln2=float((tk > LN2).mean()))
                if k == "oracle":
                    print(f"   {k:13s} {'-':24s} {'-':20s} {ppl[0]:8.2f} [{ppl[1]:.2f},{ppl[2]:.2f}] {np.median(tn):7.2f} {100 * (tn > 20).mean():8.2f}")
                else:
                    print(f"   {k:13s} {kl_[0]:6.3f} [{kl_[1]:.3f},{kl_[2]:.3f}] {top_[0]:6.1f} [{top_[1]:.1f},{top_[2]:.1f}] "
                          f"{ppl[0]:9.2f} [{ppl[1]:.1f},{ppl[2]:.1f}] {np.median(tn):7.2f} {100 * (tn > 20).mean():8.2f} "
                          f"{q[0]:6.3f}/{q[1]:5.2f}/{q[2]:5.2f} {100 * (tk > LN2).mean():8.1f}")
            res[f"{tag}_L{L}"] = row
    ctx.self_exact = True
    return res


# ------------------------------------------------------------------------------------------------ part: rollout
def part_rollout(ctx, tok, args, device, vocab):
    print("\n##### ROLLOUT  N=%d prompts, prompt %d, new %d  (current token exact; prompt exact; generated tokens' skipped-layer KV from the method)"
          % (args.rollout_n, args.rollout_prompt, args.rollout_new))
    ctx.self_exact = True
    T = args.rollout_prompt
    ch = get_data(tok, T if not args.selftest else 8, args.rollout_n, args.selftest, vocab)
    prompts = [c for c in ch]
    ref = rollout(ctx, None, 0, prompts, args.rollout_new, device)
    ref2 = rollout(ctx, None, 0, prompts[:2], args.rollout_new, device)
    assert all(bool((a == b).all()) for a, b in zip(ref[:2], ref2)), "full-model rollout not deterministic"
    ref_nll = gen_nll(ctx, prompts, ref, device)
    rng, res = np.random.RandomState(0), {"ref_nll": float(np.mean(ref_nll))}
    ks = sorted({k for k in (16, 64, 256) if k <= args.rollout_new} | {args.rollout_new})
    for L in args.depths:
        for mname, m in (("native", Native()), ("direct", Direct())):
            gen = rollout(ctx, m, L, prompts, args.rollout_new, device)
            row = {}
            for k in ks:
                mm = rollout_metrics(ref, gen, k)
                row[f"match@{k}"] = ci(mm["match_each"], rng)
            row["first_div"] = ci(mm["fd_each"], rng)
            row["gen_nll"] = ci(gen_nll(ctx, prompts, gen, device), rng)
            res[f"L{L}_{mname}"] = row
            print(f" L={L:2d} {mname:7s} " + "  ".join(f"Match@{k} {row[f'match@{k}'][0]:5.1f}±{(row[f'match@{k}'][2] - row[f'match@{k}'][1]) / 2:4.1f}" for k in ks)
                  + f"  first-div {row['first_div'][0]:5.1f}±{(row['first_div'][2] - row['first_div'][1]) / 2:4.1f}"
                  + f"  GenNLL {row['gen_nll'][0]:.3f} (full {res['ref_nll']:.3f})")
    print(" Match@k is POSITION-WISE over the first k generated tokens; first divergence = index of first mismatch (capped at the length).")
    return res


# ------------------------------------------------------------------------------------------------ part: latency
def cuda_time(fn, reps, warm=5):
    use = torch.cuda.is_available()
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        if use:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); a.record(); fn(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        else:
            t = time.perf_counter(); fn(); ts.append(1000 * (time.perf_counter() - t))
    return float(np.median(ts))


@torch.no_grad()
def part_latency(ctx, tok, args, device, vocab):
    import vkr3
    L = args.depths[0]
    dt = ctx.dtype
    print(f"\n##### MEASURED LATENCY  device={device}  dtype={dt}  L={L}/{ctx.N}  (median of {args.reps} runs, batch 1, single new token)")
    if device.type != "cuda":
        print(" ! no GPU: numbers below are CPU timings and only check the code path")
    res = {"L": L, "N": ctx.N}
    S = ctx.N - L
    kvc = KVC(ctx.d, S, kvc_rank(n_params(vkr2.VKR(ctx.d, S, 64, ctx.kvdim)), ctx.d, S, ctx.kvdim), ctx.kvdim).to(device)
    shadow = vkr3.Shadow(ctx.d, S, 16, ctx.kvdim, "", 4).to(device)
    shadowS = vkr3.Shadow(ctx.d, S, 16, ctx.kvdim, "S", 4).to(device)
    res["params"] = dict(kvc=n_params(kvc), shadow=n_params(shadow), shadowS=n_params(shadowS))

    def layer_step(i, h, cos, sin, ck, cv, p, nq=1):
        lay, a = ctx.layers[i], ctx.layers[i].self_attn
        n = lay.input_layernorm(h)
        q = ctx.rope(ctx.split(a.q_proj(n), ctx.H), cos, sin)
        k = ctx.rope(ctx.split(a.k_proj(n), ctx.Hkv), cos, sin)
        v = ctx.split(a.v_proj(n), ctx.Hkv)
        ck[i][:, :, p:p + nq] = k
        cv[i][:, :, p:p + nq] = v
        K, V = ctx.rep_kv(ck[i][:, :, :p + nq]), ctx.rep_kv(cv[i][:, :, :p + nq])
        if nq == 1:
            o = F.scaled_dot_product_attention(q, K, V)
        else:
            mask = torch.ones(nq, p + nq, dtype=torch.bool, device=h.device).tril(p)
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
        h = h + a.o_proj(o.transpose(1, 2).reshape(h.shape[0], nq, ctx.H * ctx.D))
        return h + lay.mlp(lay.post_attention_layernorm(h))

    for C in args.ctx_list:
        Tmax = C + 64
        ck = [torch.randn(1, ctx.Hkv, Tmax, ctx.D, device=device, dtype=dt) * 0.1 for _ in range(ctx.N)]
        cv = [torch.randn(1, ctx.Hkv, Tmax, ctx.D, device=device, dtype=dt) * 0.1 for _ in range(ctx.N)]
        p = C
        h0 = torch.randn(1, 1, ctx.d, device=device, dtype=dt)
        pos = torch.tensor([[p]], device=device)
        cos, sin = ctx.rotary(h0, pos)

        def f_layers(a, b, h=h0):
            x = h
            for i in range(a, b):
                x = layer_step(i, x, cos, sin, ck, cv, p)
            return x

        def f_full():
            ctx.head(f_layers(0, ctx.N))

        def f_draft():
            ctx.head(f_layers(0, L))

        def f_native():
            for i in range(L, ctx.N):
                k, v = ctx.project_kv(i, h0, cos, sin)
                ck[i][:, :, p:p + 1] = k
                cv[i][:, :, p:p + 1] = v

        def f_mod(m):
            def g():
                orc = {"hs": {L: h0}, "cos": cos, "sin": sin}
                for j, (k, v) in enumerate(m.kv(ctx, orc, L)):
                    ck[L + j][:, :, p:p + 1] = k
                    cv[L + j][:, :, p:p + 1] = v
            return g

        def f_backfill(kk):
            hk = torch.randn(1, kk, ctx.d, device=device, dtype=dt)
            posk = torch.arange(p, p + kk, device=device).view(1, -1)
            ck_, sk_ = ctx.rotary(hk, posk)

            def g():
                x = hk
                for i in range(L, ctx.N):
                    x = layer_step(i, x, ck_, sk_, ck, cv, p, nq=kk)
            return g

        r = dict(full=cuda_time(f_full, args.reps), draft=cuda_time(f_draft, args.reps), native_fill=cuda_time(f_native, args.reps),
                 kvc_fill=cuda_time(f_mod(kvc), args.reps), shadow_fill=cuda_time(f_mod(shadow), args.reps),
                 shadowS_fill=cuda_time(f_mod(shadowS), args.reps))
        for kk in (4, 16):
            r[f"backfill{kk}"] = cuda_time(f_backfill(kk), args.reps)
        res[f"ctx{C}"] = r
        print(f"\n context {C} tokens (ms per step):")
        for k, v in r.items():
            print(f"   {k:14s} {v:8.3f}")
        full = r["full"]
        print(f"   --- derived tokens/s at batch 1 (UPPER BOUNDS: every token exits at L, no verification, quality not guaranteed) ---")
        print(f"   {'full-depth':28s} {1000 / full:8.1f}   (1.00x)")
        for nm, fill in (("exit + native fill", r["native_fill"]), ("exit + KVC fill", r["kvc_fill"]), ("exit + shadowS fill", r["shadowS_fill"]),
                         ("exit + shadow fill", r["shadow_fill"])):
            t = r["draft"] + fill
            print(f"   {nm:28s} {1000 / t:8.1f}   ({full / t:.2f}x)")
        for R, key in ((4, "backfill4"),):
            for nm, fill in (("native", r["native_fill"]), ("KVC", r["kvc_fill"]), ("shadowS", r["shadowS_fill"])):
                t = (R * (r["draft"] + fill) + r[key]) / R
                print(f"   {'exit + ' + nm + ' + backfill/' + str(R):28s} {1000 / t:8.1f}   ({full / t:.2f}x)   [exact K/V restored every {R} tokens]")
        res[f"ctx{C}_derived_note"] = "upper bounds, see script docstring"
        del ck, cv
    if device.type == "cuda":
        res["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        print(f"\n peak GPU memory in this process: {res['peak_mem_mb']:.0f} MB; KV cache per token per layer: {2 * ctx.kvdim * (2 if dt != torch.float32 else 4)} bytes")
    print(" Repair-module params:", res["params"])
    return res


# ------------------------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--depths", default="18,24,30")
    p.add_argument("--part", default="all", help="fills,table1,rollout,latency or all")
    p.add_argument("--n_fill", type=int, default=60)
    p.add_argument("--n_t1", type=int, default=200)
    p.add_argument("--t1_len", type=int, default=64)
    p.add_argument("--rollout_n", type=int, default=64)
    p.add_argument("--rollout_new", type=int, default=256)
    p.add_argument("--rollout_prompt", type=int, default=32)
    p.add_argument("--ctx_list", default="512,4096")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--bs", type=int, default=6)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    parts = ["fills", "table1", "rollout", "latency"] if a.part == "all" else a.part.split(",")
    device = torch.device("cuda" if torch.cuda.is_available() and not a.selftest else "cpu")
    set_seed(0)
    if a.selftest:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=8, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=1024, rope_theta=10000.0, tie_word_embeddings=False)
        model, tok = Qwen2ForCausalLM(cfg).float().eval(), None
        a.depths, a.n_fill, a.n_t1, a.bs, a.rollout_n, a.rollout_new, a.rollout_prompt = "4,6", 8, 8, 4, 2, 12, 8
        a.ctx_list, a.reps = "32,64", 3
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        dt = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt).to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    ctx = CtxF(model)
    a.depths = [int(x) for x in a.depths.split(",")]
    a.ctx_list = [int(x) for x in a.ctx_list.split(",")]
    vocab = model.config.vocab_size
    print(f"model={a.model if not a.selftest else 'tiny'} layers={ctx.N} device={device} dtype={ctx.dtype}")
    out = {}
    for name, fn in (("fills", part_fills), ("table1", part_table1), ("rollout", part_rollout), ("latency", part_latency)):
        if name in parts:
            t0 = time.time()
            out[name] = fn(ctx, tok, a, device, vocab)
            print(f"\n[{name} done in {time.time() - t0:.0f}s]")
    fn = f"paper_fixes_{a.model.replace('/', '_') if not a.selftest else 'selftest'}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("\nwrote", fn, "- send this json back")


if __name__ == "__main__":
    main()