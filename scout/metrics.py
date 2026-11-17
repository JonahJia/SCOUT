
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

@dataclass
class PredBatch:
    probs: np.ndarray  # (N,C)
    preds: np.ndarray  # (N,)
    labels: np.ndarray  # (N,)

def classification_metrics(probs: np.ndarray, labels: np.ndarray, average: str = "macro") -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average)
    return {"acc": float(acc), "f1": float(f1)}

def auc_trapz(xs: List[float], ys: List[float]) -> float:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return float(np.trapz(ys, xs))

def deletion_insertion_auc(
    model,
    tokenizer,
    text: str,
    label: int,
    evidence_word_spans: List[Tuple[int,int]],
    max_length: int = 256,
    mode: str = "deletion",
    steps: Optional[int] = None,
    mask_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    ERASER-style deletion/insertion AUC at the word span level.

    We gradually mask (deletion) or unmask (insertion) evidence spans, recompute p(y|x),
    and compute AUC of probability vs fraction modified.

    Args:
      evidence_word_spans: list of (word_start, word_end) inclusive, ordered by importance (high->low)
      mode: "deletion" or "insertion"
    """
    assert mode in ("deletion", "insertion")

    # Word-tokenize and build a baseline masked version
    words = text.split()
    if mask_token is None:
        mask_token = tokenizer.mask_token or "[MASK]"

    # We build a list of span masks. For simplicity, spans are applied in order.
    # fraction axis uses number of spans applied / total spans.
    K = len(evidence_word_spans)
    if K == 0:
        return {"auc": None, "xs": [], "ys": []}

    if steps is None:
        steps = K

    def apply_spans(base_words: List[str], spans_to_apply: List[Tuple[int,int]], keep: bool) -> List[str]:
        out = base_words[:]
        for (l, r) in spans_to_apply:
            l = max(0, l); r = min(len(out)-1, r)
            for i in range(l, r+1):
                out[i] = out[i] if keep else mask_token
        return out

    # For insertion, start fully masked; for deletion, start original.
    if mode == "insertion":
        current = [mask_token] * len(words)
    else:
        current = words[:]

    xs, ys = [], []
    for t in range(0, steps + 1):
        frac = t / steps
        n_spans = int(round(frac * K))
        spans = evidence_word_spans[:n_spans]
        if mode == "insertion":
            current = apply_spans([mask_token]*len(words), spans, keep=True)
        else:
            current = apply_spans(words, spans, keep=False)

        x_text = " ".join(current)
        # Run model
        with torch.no_grad():
            out = model.predict_proba_texts([x_text], tokenizer=tokenizer, max_length=max_length)
            p = float(out[0, int(label)])
        xs.append(frac)
        ys.append(p)

    # For deletion, lower AUC better; for insertion, higher AUC better.
    return {"auc": auc_trapz(xs, ys), "xs": xs, "ys": ys}
