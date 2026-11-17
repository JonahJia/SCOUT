
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from scout.datasets import TextDataset, load_hf_dataset, load_local_dataset, build_collate_fn
from scout.model import SCOUT
from scout.explain import explain_case_based
from scout.render import render_cases_to_html
from scout.utils import ensure_dir, save_json, set_seed

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--anchors_path", type=str, required=True)

    ap.add_argument("--dataset", type=str, default="imdb")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--encoder", type=str, default=None, help="If not provided, read from ckpt args")
    ap.add_argument("--num_classes", type=int, default=None)

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num_cases", type=int, default=20)
    ap.add_argument("--export_json", type=str, default="Outputs/cases.json")
    ap.add_argument("--export_html", type=str, default="Outputs/cases.html")
    return ap.parse_args()

def load_data(args):
    if args.dataset == "local":
        assert args.data_dir, "--data_dir is required for local dataset"
        te_texts, te_labels = load_local_dataset(args.data_dir, args.split)
    else:
        te_texts, te_labels = load_hf_dataset(args.dataset, split=args.split)
    return te_texts, te_labels

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(os.path.dirname(args.export_json) or ".")
    ensure_dir(os.path.dirname(args.export_html) or ".")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    encoder = args.encoder or ckpt_args.get("encoder") or ckpt_args.get("encoder_name") or "roberta-base"
    num_classes = args.num_classes or ckpt_args.get("num_classes") or 2

    tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True, add_prefix_space=True)

    te_texts, te_labels = load_data(args)
    ds = TextDataset(te_texts, te_labels)
    collate = build_collate_fn(tokenizer, max_length=args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Rebuild model with ckpt hyperparams if present
    model = SCOUT(
        encoder_name=encoder,
        num_classes=num_classes,
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
        margin=float(ckpt_args.get("margin", 1.0)),
        lam_clst=float(ckpt_args.get("lam_clst", 1.0)),
        lam_sep=float(ckpt_args.get("lam_sep", 1.0)),
        lam_reg=float(ckpt_args.get("lam_reg", 0.0)),
        eta_tv=float(ckpt_args.get("eta_tv", 1.0)),
        freeze_encoder_except_last=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.load_anchors(args.anchors_path)
    model.eval()

    cases: List[Dict[str, Any]] = []
    for batch in dl:
        exps = explain_case_based(model, tokenizer, batch, top_spans=5, top_protos=3, links_per_span=5)
        cases.extend(exps)
        if len(cases) >= args.num_cases:
            cases = cases[: args.num_cases]
            break

    save_json(cases, args.export_json)
    html = render_cases_to_html(cases)
    with open(args.export_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("saved:", args.export_json)
    print("saved:", args.export_html)

if __name__ == "__main__":
    main()
