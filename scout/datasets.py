
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .utils import basic_word_tokenize

@dataclass
class TextExample:
    text: str
    label: int
    idx: int

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = [int(y) for y in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> TextExample:
        return TextExample(text=self.texts[i], label=self.labels[i], idx=i)

def load_hf_dataset(name: str, split: str = "train") -> Tuple[List[str], List[int]]:
    """
    Lightweight wrapper over HuggingFace `datasets.load_dataset`.
    Supports common text classification datasets by name.
    """
    from datasets import load_dataset  # lazy import

    ds = load_dataset(name, split=split)
    # heuristic: find text field
    text_field = None
    for cand in ["text", "sentence", "content", "review", "question", "title"]:
        if cand in ds.column_names:
            text_field = cand
            break
    if text_field is None:
        # fallback: first string field
        for col in ds.column_names:
            if isinstance(ds[0][col], str):
                text_field = col
                break
    if text_field is None:
        raise ValueError(f"Could not infer text field for dataset={name}, columns={ds.column_names}")

    label_field = "label" if "label" in ds.column_names else None
    if label_field is None:
        # fallback: first int field
        for col in ds.column_names:
            v = ds[0][col]
            if isinstance(v, int):
                label_field = col
                break
    if label_field is None:
        raise ValueError(f"Could not infer label field for dataset={name}")

    texts = [x for x in ds[text_field]]
    labels = [int(y) for y in ds[label_field]]
    return texts, labels

def load_local_dataset(
    data_dir: str,
    split: str,
    text_col: str = "text",
    label_col: str = "label",
) -> Tuple[List[str], List[int]]:
    """
    Load local dataset from CSV/TSV/JSONL.

    Expected files:
      - {split}.csv or {split}.tsv or {split}.jsonl in data_dir

    The file must contain at least text_col and label_col.
    """
    base = os.path.join(data_dir, f"{split}")
    if os.path.exists(base + ".csv"):
        path = base + ".csv"
        sep = ","
        fmt = "csv"
    elif os.path.exists(base + ".tsv"):
        path = base + ".tsv"
        sep = "\t"
        fmt = "tsv"
    elif os.path.exists(base + ".jsonl"):
        path = base + ".jsonl"
        fmt = "jsonl"
    else:
        raise FileNotFoundError(f"Cannot find {split}.csv/tsv/jsonl in {data_dir}")

    texts, labels = [], []
    if fmt in ("csv", "tsv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                texts.append(row[text_col])
                labels.append(int(row[label_col]))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                texts.append(obj[text_col])
                labels.append(int(obj[label_col]))
    return texts, labels

def build_collate_fn(tokenizer, max_length: int = 256):
    """
    Create a collate_fn that:
      1) tokenizes texts using `is_split_into_words=True` to recover word boundaries
      2) returns python lists: words, word_bounds, plus tensors: input_ids, attention_mask, labels, idx
    """
    def collate(batch: List[TextExample]) -> Dict[str, Any]:
        texts = [ex.text for ex in batch]
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
        idxs = torch.tensor([ex.idx for ex in batch], dtype=torch.long)

        words_list = [basic_word_tokenize(t) for t in texts]
        # Fast tokenizers support is_split_into_words=True and word_ids mapping.
        enc = tokenizer(
            words_list,
            is_split_into_words=True,
            return_attention_mask=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        word_bounds_batch: List[List[Tuple[int, int]]] = []
        for i in range(len(batch)):
            word_ids = enc.word_ids(batch_index=i)  # list length = seq_len
            bounds: Dict[int, List[int]] = {}
            for tok_pos, wid in enumerate(word_ids):
                if wid is None:
                    continue
                bounds.setdefault(wid, []).append(tok_pos)
            # Ensure contiguous bounds per word id
            word_bounds = [(min(pos_list), max(pos_list)) for wid, pos_list in sorted(bounds.items(), key=lambda x: x[0])]
            # The tokenization may truncate, so words_list[i] might be longer than bounds.
            # We'll keep only the words that appear in word_bounds:
            max_wid = max(bounds.keys()) if bounds else -1
            if max_wid + 1 < len(words_list[i]):
                words_list[i] = words_list[i][: max_wid + 1]
            word_bounds_batch.append(word_bounds)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "idxs": idxs,
            "words": words_list,
            "word_bounds": word_bounds_batch,
            "texts": texts,
        }

    return collate
