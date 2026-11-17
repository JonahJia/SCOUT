
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch

@dataclass
class SpanInfo:
    word_start: int
    word_end: int
    tok_start: int
    tok_end: int

def enumerate_word_spans(
    word_bounds: List[Tuple[int, int]],
    lmin: int,
    lmax: int,
) -> List[SpanInfo]:
    """
    Enumerate contiguous word spans (l,r) with length in [lmin,lmax], stride 1.
    Each word has a corresponding contiguous wordpiece token range [p_t, q_t].

    word_bounds: list of (tok_start, tok_end) per word (inclusive indices).
    """
    n_words = len(word_bounds)
    spans: List[SpanInfo] = []
    for l in range(n_words):
        for length in range(lmin, lmax + 1):
            r = l + length - 1
            if r >= n_words:
                break
            ts = word_bounds[l][0]
            te = word_bounds[r][1]
            spans.append(SpanInfo(word_start=l, word_end=r, tok_start=ts, tok_end=te))
    return spans

def span_embeddings_mean_pool(
    H: torch.Tensor,
    spans: List[SpanInfo],
) -> torch.Tensor:
    """
    Compute mean-pooled span embeddings from encoder token embeddings H.

    H: (L, D)
    spans: list of SpanInfo with token ranges [tok_start, tok_end] inclusive
    Returns: (M, D)
    """
    if len(spans) == 0:
        return H.new_zeros((0, H.size(-1)))

    L, D = H.shape
    # Prefix sum for O(1) range sums
    prefix = torch.cat([H.new_zeros((1, D)), H.cumsum(dim=0)], dim=0)  # (L+1,D)

    starts = torch.tensor([s.tok_start for s in spans], device=H.device, dtype=torch.long)
    ends = torch.tensor([s.tok_end for s in spans], device=H.device, dtype=torch.long)

    sums = prefix[ends + 1] - prefix[starts]  # (M,D)
    lens = (ends - starts + 1).clamp_min(1).unsqueeze(1).to(H.dtype)
    return sums / lens

def build_span_text(words: List[str], word_start: int, word_end: int) -> str:
    """
    Join tokens back to a readable span string.
    """
    if word_start < 0 or word_end >= len(words) or word_start > word_end:
        return ""
    # Basic join; you can replace this with a smarter detokenizer if desired.
    return " ".join(words[word_start:word_end+1])
