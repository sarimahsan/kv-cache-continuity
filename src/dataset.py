import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


def get_tokenizer():
    """
    Returns a fast BPE tokenizer (GPT-2 tokenizer from tiktoken or transformers).
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc
    except ImportError:
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained("gpt2")
        return enc


class TextDataset(Dataset):
    """
    Fixed-length chunked causal LM dataset.
    Given token IDs, slices into inputs (x) and shifted targets (y).
    """
    def __init__(self, token_ids: torch.Tensor, seq_len: int = 512):
        self.seq_len = seq_len
        # Total number of full sequences
        n_chunks = len(token_ids) // (seq_len + 1)
        if n_chunks == 0:
            # Pad if too short
            padded = torch.zeros(seq_len + 1, dtype=torch.long)
            padded[:len(token_ids)] = token_ids
            self.data = padded.unsqueeze(0)
        else:
            self.data = token_ids[:n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_dataset(
    dataset_name: str = "tinystories",
    seq_len: int = 512,
    batch_size: int = 16,
    num_workers: int = 0,
    max_train_samples: Optional[int] = 50000,
    max_val_samples: Optional[int] = 2000
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads and tokenizes a dataset for language modeling.
    Supported dataset_names:
      - 'tinystories' (roneneldan/TinyStories)
      - 'wikitext2' (wikitext-2-raw-v1)
      - 'synthetic' / fallback text corpus
    """
    tokenizer = get_tokenizer()
    encode_fn = tokenizer.encode if hasattr(tokenizer, 'encode') else (lambda s: tokenizer(s)['input_ids'])

    tokens_train = []
    tokens_val = []

    print(f"[*] Loading dataset: '{dataset_name}'...")

    try:
        if dataset_name.lower() == "tinystories":
            from datasets import load_dataset as hf_load_dataset
            raw_ds = hf_load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            
            # Stream train split
            print("[*] Streaming and tokenizing TinyStories...")
            count = 0
            for item in raw_ds:
                t = encode_fn(item['text'])
                tokens_train.extend(t)
                tokens_train.append(50256) # <|endoftext|>
                count += 1
                if max_train_samples and count >= max_train_samples:
                    break
            
            # Stream val split
            val_raw = hf_load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
            count = 0
            for item in val_raw:
                t = encode_fn(item['text'])
                tokens_val.extend(t)
                tokens_val.append(50256)
                count += 1
                if max_val_samples and count >= max_val_samples:
                    break

        elif dataset_name.lower() == "wikitext2":
            from datasets import load_dataset as hf_load_dataset
            raw_ds = hf_load_dataset("wikitext", "wikitext-2-raw-v1")
            train_text = "\n\n".join(raw_ds["train"]["text"])
            val_text = "\n\n".join(raw_ds["validation"]["text"])
            tokens_train = encode_fn(train_text)
            tokens_val = encode_fn(val_text)

        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

    except Exception as e:
        print(f"[!] Warning: Could not download '{dataset_name}' online ({e}). Falling back to local synthetic corpus.")
        # Create a rich local synthetic story corpus for instant zero-dependency testing
        synthetic_stories = [
            "Once upon a time, in a small vibrant village surrounded by emerald green hills, there lived a curious girl named Lily.",
            "Lily had a magical golden key that could open any ancient door in the enchanted library of whisperings.",
            "Every afternoon, the friendly old dragon Barnaby flew across the azure sky to teach Lily the secret language of the stars.",
            "One day, a mysterious puzzle appeared at the center of the village square, glowing with bright crystalline light.",
            "Lily and Barnaby worked together to solve the mathematical riddle of the ancient stone builders.",
            "When the final stone clicked into place, a magnificent fountain of sparkling blue water sprang from the ground.",
            "The villagers cheered and celebrated with sweet honey cakes and joyful music until the silver moon rose high in the sky.",
            "Deep in the forest of whispering pines, a tiny blue squirrel named Pip discovered an old compass pointing to hidden knowledge.",
            "Pip followed the compass across mossy rocks and babbling brooks until he reached a cave of shimmering crystals.",
            "Inside the cave, ancient inscriptions revealed how the forest and rivers lived in complete harmony for centuries.",
            "Pip carefully wrote down every word in his tiny notebook so he could share the wisdom with all the woodland animals.",
            "The wise old owl observed from a branch above and nodded with deep approval at Pip's gentle curiosity and courage."
        ] * 400

        full_text = "\n\n".join(synthetic_stories)
        encoded = encode_fn(full_text)
        split_idx = int(len(encoded) * 0.9)
        tokens_train = encoded[:split_idx]
        tokens_val = encoded[split_idx:]

    tensor_train = torch.tensor(tokens_train, dtype=torch.long)
    tensor_val = torch.tensor(tokens_val, dtype=torch.long)
    del tokens_train, tokens_val
    import gc
    gc.collect()

    print(f"[*] Dataset ready: Train tokens = {len(tensor_train):,}, Val tokens = {len(tensor_val):,}")

    train_ds = TextDataset(tensor_train, seq_len=seq_len)
    val_ds = TextDataset(tensor_val, seq_len=seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader
