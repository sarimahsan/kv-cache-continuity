"""Why do the T=64 and T=128 harnesses disagree?  (paper limitation: Key/Value split and native KL differ between harnesses)

Crosses three factors on the SAME model/depth, using only vkr2's attention-only protocol:
   self      : the current token's own K/V in skipped layers is EXACT (vkr2/vkr3 protocol)  vs  APPROXIMATE (hypothesis for the old harness)
   length    : T=128 | T=64 taken as the first half of the same T=128 samples | T=64 with its own random samples
   (sample identity is thus separated from length)
For each cell it reports  native KL, KL(K*,V^)=oracleK, KL(K^,V*)=oracleV, top-1, the oracle-substitution Value share
100*KL(oracleK)/(KL(oracleK)+KL(oracleV)) and the symmetric Shapley share  %S_V = 50*(b + c - a)/c  with
a=KL(K^,V*), b=KL(K*,V^), c=KL(K^,V^).   The cell that reproduces the old paper numbers identifies the cause.

  python harness_diag.py --model Qwen/Qwen2.5-3B --depths 18,30 --n 30
  python harness_diag.py --model meta-llama/Llama-3.2-1B --depths 8,10 --n 30
  python harness_diag.py --selftest
Needs vkr2.py beside it.   Writes harness_diag_<model>.json
"""
import argparse, json, random, sys, types
import numpy as np
import torch
import vkr2
from vkr2 import Ctx, Native, Direct, OracleK, OracleV, kl_tokens, set_seed, Data


class CtxS(Ctx):
    self_exact = True

    def layer_fwd(self, i, h, cos, sin, kv_hat=None):
        if self.self_exact or kv_hat is None:
            return super().layer_fwd(i, h, cos, sin, kv_hat)
        lay, a = self.layers[i], self.layers[i].self_attn
        B, T, _ = h.shape
        n = lay.input_layernorm(h)
        q = self.rope(self.split(a.q_proj(n), self.H), cos, sin)
        k = self.rope(self.split(a.k_proj(n), self.Hkv), cos, sin)
        v = self.split(a.v_proj(n), self.Hkv)
        khr, vhr = self.rep_kv(kv_hat[0]), self.rep_kv(kv_hat[1])
        sc = torch.matmul(q, khr.transpose(-1, -2)).float() * self.D ** -0.5
        sc = sc.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1), float("-inf"))
        A = torch.softmax(sc, dim=-1)
        out = torch.matmul(A.to(vhr.dtype), vhr).transpose(1, 2).reshape(B, T, self.H * self.D)
        h = h + a.o_proj(out)
        h = h + lay.mlp(lay.post_attention_layernorm(h))
        return h, (k, v)


@torch.no_grad()
def cell(ctx, chunks, L, bs, device):
    methods = dict(native=Native(), oracleK=OracleK(), oracleV=OracleV(), direct=Direct())
    kl = {k: [] for k in methods}
    top = {k: [] for k in methods}
    for s in range(0, len(chunks), bs):
        ids = chunks[s:s + bs].to(device)
        orc = ctx.oracle(ids)
        for name, m in methods.items():
            lg = ctx.approx(orc, L, m.kv(ctx, orc, L))[:, :-1]
            rl = orc["logits"][:, :-1]
            kl[name] += kl_tokens(rl, lg).mean(1).cpu().tolist()
            top[name] += (lg.argmax(-1) == rl.argmax(-1)).float().mean(1).cpu().tolist()
    r = {k: float(np.mean(v)) for k, v in kl.items()}
    a, b, c = r["oracleV"], r["oracleK"], r["native"]
    rng = np.random.RandomState(0)
    x = np.array(kl["native"])
    bs_ = x[rng.randint(0, len(x), (2000, len(x)))].mean(1)
    return dict(native=c, oracleK=b, oracleV=a, direct=r["direct"], top1=100 * float(np.mean(top["native"])),
                value_share=100 * b / (a + b), shapley_V=100 * 0.5 * (b + c - a) / c,
                native_ci=[float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5))], n=len(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--depths", default="18,30")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--bs", type=int, default=6)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not a.selftest else "cpu")
    if a.selftest:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=8, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False)
        model, tok = Qwen2ForCausalLM(cfg).float().eval(), None
        a.depths, a.n, a.bs = "4,6", 8, 4
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        dt = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt).to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    ctx = CtxS(model)
    vocab = model.config.vocab_size

    def data(T):
        ns = types.SimpleNamespace(seq_len=T, eval_n=a.n, selftest=a.selftest, no_c4=True, c4_docs=0, seed=0)
        return Data(tok, ns, vocab, False).eval["wikitext"]

    set_seed(0)
    d128, d64 = data(128 if not a.selftest else 48), data(64 if not a.selftest else 24)
    half = (64 if not a.selftest else 24)
    lengths = {"T128": d128, "T64_firsthalf_of_same": d128[:, :half], "T64_own_samples": d64}
    out = {}
    for L in [int(x) for x in a.depths.split(",")]:
        print(f"\n=== {a.model}  L={L}/{ctx.N}  N={a.n} ===")
        print(f"{'self':7s} {'length':24s} {'native':>7s} {'oracleK':>8s} {'oracleV':>8s} {'top1':>6s} {'Vshare':>7s} {'Shapley':>8s}")
        for self_exact in (True, False):
            ctx.self_exact = self_exact
            for name, ch in lengths.items():
                r = cell(ctx, ch, L, a.bs, device)
                out[f"L{L}|self_{'exact' if self_exact else 'approx'}|{name}"] = r
                print(f"{'exact' if self_exact else 'approx':7s} {name:24s} {r['native']:7.3f} {r['oracleK']:8.3f} {r['oracleV']:8.3f} "
                      f"{r['top1']:6.1f} {r['value_share']:7.1f} {r['shapley_V']:8.1f}")
    fn = f"harness_diag_{a.model.replace('/', '_') if not a.selftest else 'selftest'}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"\nwrote {fn}\nPaper (T=64, N=30) targets: Qwen-3B L18 native 0.914 (Shapley 70.3), L30 Shapley 75.8; Llama L8 native 0.590 Shapley 76.2, L10 Shapley 82.1; Qwen-7B L14 native 1.425.")
    print("Read: if 'approx' + T64 matches the paper while 'exact' matches the repair harness, the current-token K/V handling is the cause.")


if __name__ == "__main__":
    main()