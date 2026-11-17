
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .spans import build_span_text

def explain_case_based(
    model,
    tokenizer,
    batch: Dict[str, Any],
    top_spans: int = 5,
    top_protos: int = 3,
    links_per_span: int = 5,
    use_predicted_class: bool = True,
) -> List[Dict[str, Any]]:
    """
    Produce case-based explanations for a batch.

    Output per example:
      - pred/probs
      - evidence spans (top by transported mass)
      - top responsible prototypes pi
      - anchors for those prototypes (from model.anchor_db)
      - links from evidence spans -> prototype supports
    """
    device = next(model.parameters()).device
    out = model.forward(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        word_bounds=batch["word_bounds"],
        words=batch["words"],
        # Evidence and prototype responsibilities must refer to the same class.
        labels=None if use_predicted_class or batch.get("labels") is None else batch["labels"].to(device),
        return_transport=True,
    )

    probs = out.probs.detach().cpu()
    preds = probs.argmax(dim=1).tolist()
    labels = batch["labels"].tolist() if "labels" in batch and batch["labels"] is not None else [None] * len(preds)

    # Build fast anchor lookup: anchors[class][proto] -> list of supports
    anchors = {}
    for rec in getattr(model, "anchor_db", []) or []:
        c = int(rec["class"]); k = int(rec["prototype"])
        anchors.setdefault(c, {}).setdefault(k, []).append(rec)
    for c in anchors:
        for k in anchors[c]:
            anchors[c][k].sort(key=lambda r: r.get("support", 0))

    results: List[Dict[str, Any]] = []
    for i in range(len(preds)):
        y = preds[i] if use_predicted_class else labels[i]
        if y is None:
            y = preds[i]

        pi_y = out.pi[i, y].detach().cpu()  # (K,)
        topk = torch.topk(pi_y, k=min(top_protos, pi_y.numel()), largest=True)
        top_proto_ids = topk.indices.tolist()
        top_proto_scores = topk.values.tolist()

        per = out.aux["per_example"][i]
        evidence = per.get("evidence", [])[:top_spans]
        links = per.get("links", [])
        # truncate links per span
        for item in links:
            item["top"] = item["top"][:links_per_span]

        proto_items = []
        for pid, score in zip(top_proto_ids, top_proto_scores):
            proto_items.append({
                "prototype": int(pid),
                "responsibility": float(score),
                "anchors": anchors.get(int(y), {}).get(int(pid), []),
            })

        results.append({
            "idx": int(batch["idxs"][i].item()) if "idxs" in batch else i,
            "text": batch["texts"][i] if "texts" in batch else None,
            "gold": labels[i],
            "pred": preds[i],
            "probs": probs[i].tolist(),
            "explained_class": int(y),
            "evidence_spans": evidence,
            "top_prototypes": proto_items,
            "span_support_links": links,
        })

    return results
