#!/usr/bin/env python3
"""
Backward-compatibility alias for vkr3_multi_angle_repair.py.
Canonical implementation is located at: experiments/vkr3_multi_angle_repair.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vkr3_multi_angle_repair import *

if __name__ == "__main__":
    main()
