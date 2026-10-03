"""
src/eval_dataset.py

Robust WikiText-2 Dataset Manager for Authentic Sequence Evaluation & Adapter Training.
Ensures teacher-forced evaluations and exit adapters are trained and tested on genuine,
disjoint WikiText-2 splits (train vs. validation) rather than tiny hand-picked memorization sets.
"""

import os
import math
from typing import List, Generator, Tuple, Optional
import torch
import torch.nn.functional as F

WIKITEXT2_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wikitext2")


def load_raw_wikitext2_split(split: str = "validation") -> str:
    """
    Loads raw text for WikiText-2 from Hugging Face datasets with robust error handling.
    split: 'train', 'validation', or 'test'.
    """
    # 1. Try Hugging Face datasets with canonical namespace 'Salesforce/wikitext'
    try:
        from datasets import load_dataset
        for repo_id in ["Salesforce/wikitext", "wikitext"]:
            try:
                ds = load_dataset(repo_id, "wikitext-2-raw-v1", split=split)
                texts = [item["text"] for item in ds if len(item["text"].strip()) > 0]
                full_text = "\n\n".join(texts)
                if len(full_text) > 1000:
                    return full_text
            except Exception:
                continue
    except Exception as e:
        print(f"[!] Warning: HF datasets loading failed: {e}")

    # 2. Try direct parquet download via pandas if datasets failed
    try:
        import pandas as pd
        split_name = "validation" if split == "validation" else ("train" if split == "train" else "test")
        parquet_url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{split_name}-00000-of-00001.parquet"
        print(f"[*] Attempting direct parquet download from {parquet_url}...")
        df = pd.read_parquet(parquet_url)
        texts = [t for t in df["text"].tolist() if isinstance(t, str) and len(t.strip()) > 0]
        full_text = "\n\n".join(texts)
        if len(full_text) > 1000:
            return full_text
    except Exception as e:
        print(f"[!] Parquet fallback failed: {e}")

    # 3. Check if local fallback text exists
    os.makedirs(WIKITEXT2_CACHE_DIR, exist_ok=True)
    local_txt = os.path.join(WIKITEXT2_CACHE_DIR, f"wikitext2_{split}.txt")
    if os.path.exists(local_txt):
        print(f"[*] Reading cached local text file: {local_txt}")
        with open(local_txt, "r", encoding="utf-8") as f:
            return f.read()

    raise RuntimeError(
        f"Cannot load WikiText-2 split '{split}'. Please ensure internet access or provide {local_txt}."
    )


def get_tokenized_wikitext2(
    tokenizer,
    split: str = "validation",
    cache_dir: str = WIKITEXT2_CACHE_DIR
) -> torch.Tensor:
    """
    Returns a 1D torch.LongTensor of token IDs for the specified split.
    Caches tokenized tensors to disk to avoid repeated tokenization overhead.
    """
    os.makedirs(cache_dir, exist_ok=True)
    model_name = getattr(tokenizer, "name_or_path", "unknown").replace("/", "_")
    cache_file = os.path.join(cache_dir, f"wikitext2_{split}_{model_name}.pt")

    if os.path.exists(cache_file):
        print(f"[*] Loading cached tokenized WikiText-2 '{split}' from {cache_file}...")
        return torch.load(cache_file, map_location="cpu")

    print(f"[*] Tokenizing WikiText-2 '{split}' split for {model_name}...")
    raw_text = load_raw_wikitext2_split(split)

    if hasattr(tokenizer, "encode"):
        toks = tokenizer.encode(raw_text)
        if not isinstance(toks, torch.Tensor):
            toks = torch.tensor(toks, dtype=torch.long)
    else:
        toks = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    print(f"✓ Tokenized {split} split: {len(toks):,} tokens.")
    torch.save(toks, cache_file)
    return toks


def get_wikitext2_eval_sequences(
    tokenizer,
    split: str = "validation",
    seq_len: int = 512,
    num_samples: Optional[int] = 200,
    device: torch.device = torch.device("cpu")
) -> List[torch.Tensor]:
    """
    Chunks the WikiText-2 split into non-overlapping sequences of length seq_len.
    Returns a list of tensors of shape (1, seq_len).
    """
    toks = get_tokenized_wikitext2(tokenizer, split=split)
    total_tokens = len(toks)
    num_chunks = total_tokens // seq_len

    if num_samples is not None:
        num_chunks = min(num_chunks, num_samples)

    print(f"[*] Extracting {num_chunks} non-overlapping sequences (seq_len={seq_len}) from '{split}'...")
    sequences = []
    for i in range(num_chunks):
        chunk = toks[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
        sequences.append(chunk)

    return sequences


def create_wikitext2_train_batches(
    tokenizer,
    batch_size: int = 4,
    seq_len: int = 512,
    device: torch.device = torch.device("cpu")
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Yields training batches (inputs, targets) of shape (B, seq_len) indefinitely
    from the WikiText-2 *train* split.
    """
    toks = get_tokenized_wikitext2(tokenizer, split="train").to(device)
    total_tokens = len(toks)
    offset = 0

    while True:
        batch_inputs = []
        batch_targets = []
        for _ in range(batch_size):
            if offset + seq_len + 1 > total_tokens:
                offset = 0
            chunk = toks[offset : offset + seq_len + 1]
            if len(chunk) < seq_len + 1:
                chunk = F.pad(chunk, (0, seq_len + 1 - len(chunk)), value=0)
            batch_inputs.append(chunk[:-1])
            batch_targets.append(chunk[1:])
            offset = (offset + seq_len) % (total_tokens - seq_len - 1)

        x_batch = torch.stack(batch_inputs, dim=0)
        y_batch = torch.stack(batch_targets, dim=0)
        yield x_batch, y_batch
