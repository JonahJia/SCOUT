
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def basic_word_tokenize(text: str) -> List[str]:
    """
    Tokenize to a word sequence for span enumeration.

    We keep punctuation as separate tokens to avoid losing evidence boundaries.
    This is *not* meant to be linguistically perfect; it is deterministic and
    works with `tokenizer(..., is_split_into_words=True)` for fast tokenizers.
    """
    text = text.strip()
    if not text:
        return []
    return _WORD_RE.findall(text)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@dataclass
class AnchorRecord:
    cls: int
    proto: int
    support: int
    train_id: int
    word_start: int
    word_end: int
    span_text: str
    full_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.cls,
            "prototype": self.proto,
            "support": self.support,
            "train_id": self.train_id,
            "word_start": self.word_start,
            "word_end": self.word_end,
            "span_text": self.span_text,
            "full_text": self.full_text,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AnchorRecord":
        return AnchorRecord(
            cls=int(d["class"]),
            proto=int(d["prototype"]),
            support=int(d["support"]),
            train_id=int(d["train_id"]),
            word_start=int(d["word_start"]),
            word_end=int(d["word_end"]),
            span_text=str(d["span_text"]),
            full_text=str(d["full_text"]),
        )
