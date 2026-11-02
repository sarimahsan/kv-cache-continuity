#!/usr/bin/env python3
"""
train_cheap_pass.py -- Part 1: Train Cheap-Pass KV Adapters and run Offline Fidelity Gate Check.
Saves checkpoints (adapter_E{E}_r{rank}_primary.pt) for later evaluation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cheap_pass

if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "train"])
    cheap_pass.main()

