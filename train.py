
from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from scout.datasets import TextDataset, load_hf_dataset, load_local_dataset, build_collate_fn
from scout.model import SCOUT
from scout.utils import ensure_dir, set_seed
from scout.metrics import classification_metrics

torch.autograd.set_detect_anomaly(True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="imdb", help="HuggingFace dataset name (e.g., imdb, yelp_polarity, amazon_polarity, dbpedia_14) or 'local'")
    ap.add_argument("--data_dir", type=str, default=None, help="For --dataset local: directory with train/test(/val) files.")
    ap.add_argument("--encoder", type=str, default="roberta-base")
    ap.add_argument("--num_classes", type=int, default=2)

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    # SCOUT hyperparams
    ap.add_argument("--protos_per_class", type=int, default=5)
    ap.add_argument("--proto_supports", type=int, default=5)
    ap.add_argument("--span_lmin", type=int, default=1)
    ap.add_argument("--span_lmax", type=int, default=5)
    ap.add_argument("--span_top_n", type=int, default=50)

    ap.add_argument("--uot_eps", type=float, default=0.05)
    ap.add_argument("--uot_rho", type=float, default=1.0)
    ap.add_argument("--uot_iters", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=10.0)
    ap.add_argument("--beta", type=float, default=1.0)

    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lam_clst", type=float, default=1.0)
    ap.add_argument("--lam_sep", type=float, default=1.0)
    ap.add_argument("--lam_reg", type=float, default=0.1)  # set >0 if you implement exact reg on transport
    ap.add_argument("--eta_tv", type=float, default=1.0)

    ap.add_argument("--models_dir", type=str, default="Models")
    ap.add_argument("--anchors_path", type=str, default="Models/anchors.json")
    ap.add_argument("--ckpt_path", type=str, default="Models/best.pt")
    return ap.parse_args()

def load_data(args):
    if args.dataset == "local":
        assert args.data_dir, "--data_dir is required for local dataset"
        tr_texts, tr_labels = load_local_dataset(args.data_dir, "train")
        te_texts, te_labels = load_local_dataset(args.data_dir, "test")
        # optional val
        try:
            va_texts, va_labels = load_local_dataset(args.data_dir, "val")
        except FileNotFoundError:
            # simple split from train
            n = len(tr_texts)
            cut = int(0.9 * n)
            va_texts, va_labels = tr_texts[cut:], tr_labels[cut:]
            tr_texts, tr_labels = tr_texts[:cut], tr_labels[:cut]
    else:
        tr_texts, tr_labels = load_hf_dataset(args.dataset, split="train")
        # hf datasets often have a test split
        try:
            te_texts, te_labels = load_hf_dataset(args.dataset, split="test")
        except Exception:
            # fallback: split train
            n = len(tr_texts)
            cut = int(0.9 * n)
            te_texts, te_labels = tr_texts[cut:], tr_labels[cut:]
            tr_texts, tr_labels = tr_texts[:cut], tr_labels[:cut]
        # val split if present
        try:
            va_texts, va_labels = load_hf_dataset(args.dataset, split="validation")
        except Exception:
            # split from train
            n = len(tr_texts)
            cut = int(0.9 * n)
            va_texts, va_labels = tr_texts[cut:], tr_labels[cut:]
            tr_texts, tr_labels = tr_texts[:cut], tr_labels[:cut]

    return (tr_texts, tr_labels), (va_texts, va_labels), (te_texts, te_labels)

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in dl:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            word_bounds=batch["word_bounds"],
            words=batch["words"],
            labels=None,
            return_transport=False,
        )
        all_probs.append(out.probs.detach().cpu())
        all_labels.append(batch["labels"].cpu())
    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    model.train()
    return classification_metrics(probs, labels, average="macro")

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.models_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True, add_prefix_space=True)

    (tr_texts, tr_labels), (va_texts, va_labels), (te_texts, te_labels) = load_data(args)
    train_ds = TextDataset(tr_texts, tr_labels)
    val_ds = TextDataset(va_texts, va_labels)
    test_ds = TextDataset(te_texts, te_labels)

    collate = build_collate_fn(tokenizer, max_length=args.max_length)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = SCOUT(
        encoder_name=args.encoder,
        num_classes=args.num_classes,
        protos_per_class=args.protos_per_class,
        proto_supports=args.proto_supports,
        span_lmin=args.span_lmin, span_lmax=args.span_lmax, span_top_n=args.span_top_n,
        max_length=args.max_length,
        uot_eps=args.uot_eps, uot_rho=args.uot_rho, uot_iters=args.uot_iters,
        gamma=args.gamma, beta=args.beta,
        margin=args.margin,
        lam_clst=args.lam_clst, lam_sep=args.lam_sep, lam_reg=args.lam_reg,
        eta_tv=args.eta_tv,
    ).to(device)

    # Initialize prototypes from training data (anchored)
    model.initialize_prototypes_from_data(train_dl, device=device)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    best = None
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch}")
        for batch in pbar:
            optim.zero_grad(set_to_none=True)
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                word_bounds=batch["word_bounds"],
                words=batch["words"],
                labels=batch["labels"].to(device),
                return_transport=False,  # set True if you implement exact LReg with transports (slower)
            )
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            pbar.set_postfix({"loss": float(loss.item())})

        # Projection (Readable Anchoring) at end of epoch
        model.project_prototypes(train_dl, device=device)

        val_metrics = evaluate(model, val_dl, device=device)
        print(f"[epoch {epoch}] val:", val_metrics)

        score = val_metrics["f1"]
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": epoch, "val": val_metrics}
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, args.ckpt_path)
            # Keep readable anchors aligned with the selected checkpoint.
            model.save_anchors(args.anchors_path)
            print(f"  saved best -> {args.ckpt_path}")

    # Final test eval
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_dl, device=device)
    print("[best] val:", best)
    print("[test]:", test_metrics)
    print("anchors saved:", args.anchors_path)

if __name__ == "__main__":
    main()
