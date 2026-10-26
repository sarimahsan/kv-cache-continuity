"""Speedup of draft-at-exit self-speculative decoding from a measured draft acceptance (corrected cost model).
speedup(gamma) = E[tokens/round] / (1 + gamma*c),  E = (1 - a^(gamma+1)) / (1 - a),  c = exit_depth / total_depth.
Usage: python spec_cost.py results_v3.json [--layers 36]
"""
import json, sys
f = sys.argv[1]; N = int(sys.argv[sys.argv.index("--layers") + 1]) if "--layers" in sys.argv else 36
rep = json.load(open(f))["report"]
for L, r in rep.items():
    c = int(L) / N
    for dom, ms in r.get("main", {}).items():
        for name, v in ms.items():
            if isinstance(v, dict) and "exit_top1" in v:
                a = v["exit_top1"] / 100
                row = []
                for g in (1, 2, 3, 4):
                    E = (1 - a ** (g + 1)) / (1 - a)
                    row.append(f"g{g}={E / (1 + g * c):.2f}x")
                print(f"L={L} c={c:.2f} {dom:9s} {name:9s} accept={a:.2f}  " + "  ".join(row))
print("\nbreak-even accept at gamma=1 is a > c;  ceiling for any a is about 1/c")