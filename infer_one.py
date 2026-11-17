
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from scout.model import SCOUT
from scout.datasets import TextDataset, build_collate_fn
from scout.explain import explain_case_based
from scout.render import render_cases_to_html

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--anchors_path", type=str, required=True)
    ap.add_argument("--encoder", type=str, required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--out_html", type=str, default="Outputs/one.html")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True, add_prefix_space=True)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = SCOUT(
        encoder_name=args.encoder,
        num_classes=args.num_classes,
        protos_per_class=int(ckpt_args.get("protos_per_class", 5)),
        proto_supports=int(ckpt_args.get("proto_supports", 5)),
        span_lmin=int(ckpt_args.get("span_lmin", 1)),
        span_lmax=int(ckpt_args.get("span_lmax", 5)),
        span_top_n=int(ckpt_args.get("span_top_n", 50)),
        max_length=int(ckpt_args.get("max_length", args.max_length)),
        uot_eps=float(ckpt_args.get("uot_eps", 0.05)),
        uot_rho=float(ckpt_args.get("uot_rho", 1.0)),
        uot_iters=int(ckpt_args.get("uot_iters", 20)),
        gamma=float(ckpt_args.get("gamma", 10.0)),
        beta=float(ckpt_args.get("beta", 1.0)),
        freeze_encoder_except_last=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.load_anchors(args.anchors_path)
    model.eval()

    ds = TextDataset([args.text], [0])
    collate = build_collate_fn(tokenizer, max_length=args.max_length)
    batch = collate([ds[0]])

    exps = explain_case_based(model, tokenizer, batch, top_spans=5, top_protos=3, links_per_span=5)
    html = render_cases_to_html(exps, title="SCOUT single-text explanation")

    os.makedirs(os.path.dirname(args.out_html) or ".", exist_ok=True)
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("saved:", args.out_html)
    print("prediction:", exps[0]["pred"], "probs:", exps[0]["probs"])

if __name__ == "__main__":
    main()
