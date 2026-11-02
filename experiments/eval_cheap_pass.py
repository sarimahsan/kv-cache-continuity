#!/usr/bin/env python3
"""
eval_cheap_pass.py -- Part 2: Evaluate Attention-Only Test KL & Pre-Registered Gate Check.
Loads saved adapter checkpoints and benchmarks against Oracle, NativeProj, and ZeroInit.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cheap_pass

if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "eval"])
    cheap_pass.main()
