from dataclasses import dataclass
from typing import Optional
from src.model import ModelConfig


def get_model_preset(preset_name: str) -> ModelConfig:
    """
    Returns standardized model architecture presets for research scaling:
      - '25M': Discovery Model (8 layers, d=384, 6 heads, ~25M params)
      - '50M': Main Study Model (10 layers, d=512, 8 heads, ~50M params)
      - '125M': Scaling Validation Model (12 layers, d=768, 12 heads, ~125M params)
    """
    name = preset_name.upper().strip()
    if name == "25M":
        return ModelConfig(
            vocab_size=50257,
            n_layers=8,
            d_model=384,
            n_heads=6,
            d_ff=1536,
            max_seq_len=256,
            dropout=0.0
        )
    elif name == "50M":
        return ModelConfig(
            vocab_size=50257,
            n_layers=10,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            max_seq_len=256,
            dropout=0.0
        )
    elif name == "125M":
        return ModelConfig(
            vocab_size=50257,
            n_layers=12,
            d_model=768,
            n_heads=12,
            d_ff=3072,
            max_seq_len=512,
            dropout=0.0
        )
    else:
        raise ValueError(f"Unknown preset '{preset_name}'. Choose from ['25M', '50M', '125M'].")
