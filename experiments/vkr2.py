#!/usr/bin/env python3
"""
Backward-compatibility alias for vkr2_unified_repair.py.
Canonical implementation is located at: experiments/vkr2_unified_repair.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vkr2_unified_repair import *

if __name__ == "__main__":
    main()
