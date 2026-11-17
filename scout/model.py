
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from .spans import enumerate_word_spans, span_embeddings_mean_pool, build_span_text, SpanInfo
from .uot import uot_sinkhorn_log, uot_objective
from .utils import AnchorRecord

@dataclass
class SCOUTOutputs:
    logits: torch.Tensor              # (B,C)
    probs: torch.Tensor               # (B,C)
    D: torch.Tensor                   # (B,C) class distances
    d_ck: torch.Tensor                # (B,C,K) proto distances
    pi: torch.Tensor                  # (B,C,K) responsibilities
    loss: Optional[torch.Tensor] = None
    aux: Optional[Dict[str, Any]] = None

class SCOUT(nn.Module):
    """
    Standalone SCOUT implementation.

    Core paper mapping:
    - span distribution μx: build_spans() / scoring head (Sec. 3.1)
    - anchored prototypes νc,k: proto_supports buffer + projection (Sec. 3.2)
    - UOT layer: log-domain generalized Sinkhorn (Sec. 3.3, Alg. 1)
    - soft-min aggregation & distance-softmax classifier (Sec. 3.4)
    - loss: CE + Clst + Sep + Reg (Sec. 3.5)
    """
    def __init__(
        self,
        encoder_name: str,
        num_classes: int,
        protos_per_class: int = 5,
        proto_supports: int = 5,
        span_lmin: int = 1,
        span_lmax: int = 5,
        span_top_n: int = 50,
        max_length: int = 256,
        uot_eps: float = 0.05,
        uot_rho: float = 1.0,
        uot_iters: int = 20,
        gamma: float = 10.0,
        beta: float = 1.0,
        margin: float = 1.0,
        lam_clst: float = 1.0,
        lam_sep: float = 1.0,
        lam_reg: float = 0.1,
        eta_tv: float = 1.0,
        freeze_encoder_except_last: bool = True,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.K = protos_per_class
        self.Mp = proto_supports
        self.span_lmin = span_lmin
        self.span_lmax = span_lmax
        self.span_top_n = span_top_n
        self.max_length = max_length

        self.uot_eps = float(uot_eps)
        self.uot_rho = float(uot_rho)
        self.uot_iters = int(uot_iters)

        self.gamma = float(gamma)
        self.beta = float(beta)

        self.margin = float(margin)
        self.lam_clst = float(lam_clst)
        self.lam_sep = float(lam_sep)
        self.lam_reg = float(lam_reg)
        self.eta_tv = float(eta_tv)

        # Encoder
        config = AutoConfig.from_pretrained(encoder_name)
        base = AutoModel.from_pretrained(encoder_name, config=config)
        # If encoder-decoder, use encoder only
        self.encoder = getattr(base, "get_encoder", lambda: base)()
        self.hidden_size = getattr(config, "hidden_size", None) or getattr(config, "d_model")

        # span scoring head g(.)
        self.span_scorer = nn.Linear(self.hidden_size, 1)

        # Anchored prototype supports: (C,K,Mp,D)
        self.register_buffer("proto_supports", torch.zeros(num_classes, self.K, self.Mp, self.hidden_size))
        # Anchor metadata db: list of AnchorRecord dictionaries
        self.anchor_db: List[Dict[str, Any]] = []

        if freeze_encoder_except_last:
            self._freeze_encoder_except_last_layer()

    def _freeze_encoder_except_last_layer(self) -> None:
        """
        Freeze encoder parameters except the last transformer block (best-effort across architectures).
        """
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Common patterns:
        # BERT/RoBERTa/DeBERTa: encoder.layer[-1]
        # DistilBERT: transformer.layer[-1]
        candidates = []
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            candidates.append(self.encoder.encoder.layer)
        if hasattr(self.encoder, "layer"):
            candidates.append(self.encoder.layer)
        if hasattr(self.encoder, "transformer") and hasattr(self.encoder.transformer, "layer"):
            candidates.append(self.encoder.transformer.layer)

        last_layer = None
        for layers in candidates:
            if isinstance(layers, (list, nn.ModuleList)) and len(layers) > 0:
                last_layer = layers[-1]
                break

        if last_layer is None:
            # fallback: unfreeze all
            for p in self.encoder.parameters():
                p.requires_grad = True
            return

        for p in last_layer.parameters():
            p.requires_grad = True

    def has_anchors(self) -> bool:
        return self.anchor_db is not None and len(self.anchor_db) > 0

    def save_anchors(self, path: str) -> None:
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.anchor_db, f, ensure_ascii=False, indent=2)

    def load_anchors(self, path: str) -> None:
        import json
        with open(path, "r", encoding="utf-8") as f:
            self.anchor_db = json.load(f)

    @torch.no_grad()
    def initialize_prototypes_from_data(
        self,
        train_loader,
        device: torch.device,
        max_pool_per_class: int = 2000,
    ) -> None:
        """
        Initialize proto_supports by sampling real spans from training data (recommended).
        This produces anchored prototypes from the start.

        We collect up to max_pool_per_class candidate spans per class (top-N per example),
        then sample K*Mp supports uniformly.
        """
        self.eval()
        pools = {c: [] for c in range(self.num_classes)}
        meta = {c: [] for c in range(self.num_classes)}

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            words_list = batch["words"]
            word_bounds = batch["word_bounds"]
            idxs = batch["idxs"].tolist()
            texts = batch["texts"]

            H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B,L,D)

            for i in range(H.size(0)):
                c = int(labels[i].item())
                if len(pools[c]) >= max_pool_per_class:
                    continue
                spans = enumerate_word_spans(word_bounds[i], self.span_lmin, self.span_lmax)
                if len(spans) == 0:
                    continue
                emb = span_embeddings_mean_pool(H[i], spans)  # (M,D)
                scores = self.span_scorer(emb).squeeze(-1)  # (M,)
                # keep top span_top_n candidates
                topn = min(self.span_top_n, emb.size(0))
                vals, idx_top = torch.topk(scores, k=topn, largest=True)
                for j in idx_top.tolist():
                    pools[c].append(emb[j].detach().cpu())
                    s = spans[j]
                    meta[c].append((idxs[i], s.word_start, s.word_end, texts[i], words_list[i]))

            # early stop if all pools full
            if all(len(pools[c]) >= max_pool_per_class for c in range(self.num_classes)):
                break

        # Sample supports
        self.anchor_db = []
        for c in range(self.num_classes):
            candidates = pools[c]
            if len(candidates) == 0:
                raise RuntimeError(f"No candidate spans collected for class {c}. Check dataset/labels.")
            cand_tensor = torch.stack(candidates, dim=0)  # (N,D)
            N = cand_tensor.size(0)
            need = self.K * self.Mp
            # sample with replacement if N < need
            idx_sample = torch.randint(low=0, high=N, size=(need,))
            sampled = cand_tensor[idx_sample].to(device)
            self.proto_supports[c].copy_(sampled.view(self.K, self.Mp, -1))

            # Fill anchor db with sampled metadata
            for kk in range(self.K):
                for jj in range(self.Mp):
                    si = idx_sample[kk * self.Mp + jj].item()
                    train_id, ws, we, full_text, words = meta[c][si]
                    span_text = build_span_text(words, ws, we)
                    self.anchor_db.append(
                        AnchorRecord(
                            cls=c, proto=kk, support=jj,
                            train_id=int(train_id), word_start=int(ws), word_end=int(we),
                            span_text=span_text, full_text=str(full_text)
                        ).to_dict()
                    )

        self.train()

    @torch.no_grad()
    def project_prototypes(
        self,
        train_loader,
        device: torch.device,
        max_pool_per_class: int = 5000,
        chunk: int = 1024,
    ) -> None:
        """
        Readable Anchoring projection step (Eq. (5)).
        At end of each epoch, replace each support by its nearest neighbor span in class-c training data.

        This updates:
          - self.proto_supports (values)
          - self.anchor_db (metadata)
        """
        self.eval()
        pools = {c: [] for c in range(self.num_classes)}
        meta = {c: [] for c in range(self.num_classes)}  # (train_id, ws, we, full_text, words)

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            words_list = batch["words"]
            word_bounds = batch["word_bounds"]
            idxs = batch["idxs"].tolist()
            texts = batch["texts"]

            H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B,L,D)

            for i in range(H.size(0)):
                c = int(labels[i].item())
                if len(pools[c]) >= max_pool_per_class:
                    continue
                spans = enumerate_word_spans(word_bounds[i], self.span_lmin, self.span_lmax)
                if len(spans) == 0:
                    continue
                emb = span_embeddings_mean_pool(H[i], spans)  # (M,D)
                scores = self.span_scorer(emb).squeeze(-1)  # (M,)
                topn = min(self.span_top_n, emb.size(0))
                vals, idx_top = torch.topk(scores, k=topn, largest=True)
                for j in idx_top.tolist():
                    pools[c].append(emb[j].detach().cpu())
                    s = spans[j]
                    meta[c].append((idxs[i], s.word_start, s.word_end, texts[i], words_list[i]))

            if all(len(pools[c]) >= max_pool_per_class for c in range(self.num_classes)):
                break

        new_anchor_db: List[Dict[str, Any]] = []
        for c in range(self.num_classes):
            candidates = pools[c]
            if len(candidates) == 0:
                continue
            cand = torch.stack(candidates, dim=0).to(device)  # (N,D)
            N = cand.size(0)

            supports = self.proto_supports[c].view(self.K * self.Mp, -1)  # (K*Mp,D)
            # Find nearest neighbor in chunks to reduce memory if N large
            best_idx = []
            best_dist = []
            for s0 in range(0, supports.size(0), chunk):
                s1 = min(s0 + chunk, supports.size(0))
                sup_chunk = supports[s0:s1]  # (cN,D)
                d = torch.cdist(sup_chunk, cand)  # (cN,N)
                vals, idxs_nn = torch.min(d, dim=1)
                best_idx.append(idxs_nn)
                best_dist.append(vals)
            best_idx = torch.cat(best_idx, dim=0)  # (K*Mp,)
            # update supports
            chosen = cand[best_idx]  # (K*Mp,D)
            self.proto_supports[c].copy_(chosen.view(self.K, self.Mp, -1))

            # record metadata
            for kk in range(self.K):
                for jj in range(self.Mp):
                    si = int(best_idx[kk * self.Mp + jj].item())
                    train_id, ws, we, full_text, words = meta[c][si]
                    span_text = build_span_text(words, ws, we)
                    new_anchor_db.append(
                        AnchorRecord(
                            cls=c, proto=kk, support=jj,
                            train_id=int(train_id), word_start=int(ws), word_end=int(we),
                            span_text=span_text, full_text=str(full_text)
                        ).to_dict()
                    )

        self.anchor_db = new_anchor_db
        self.train()

    def _build_span_distribution(
        self,
        H_i: torch.Tensor,                   # (L,D)
        words: List[str],
        word_bounds: List[Tuple[int,int]],
    ) -> Dict[str, Any]:
        spans = enumerate_word_spans(word_bounds, self.span_lmin, self.span_lmax)
        if len(spans) == 0:
            return {
                "spans": [],
                "emb": H_i.new_zeros((0, H_i.size(-1))),
                "a": H_i.new_zeros((0,)),
            }
        emb = span_embeddings_mean_pool(H_i, spans)  # (M,D)
        scores = self.span_scorer(emb).squeeze(-1)  # (M,)
        a = torch.softmax(scores, dim=0)  # (M,)

        # Keep top-N spans to control OT complexity
        if self.span_top_n is not None and self.span_top_n > 0 and emb.size(0) > self.span_top_n:
            vals, idx_top = torch.topk(a, k=self.span_top_n, largest=True)
            emb = emb[idx_top]
            a = vals / vals.sum().clamp_min(1e-8)
            spans = [spans[j] for j in idx_top.tolist()]

        return {"spans": spans, "emb": emb, "a": a}

    def _cosine_cost(self, X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Cost matrix C_ij = 1 - cos(x_i, y_j)
        X: (M,D), Y: (Mp,D)
        """
        Xn = X / X.norm(dim=1, keepdim=True).clamp_min(eps)
        Yn = Y / Y.norm(dim=1, keepdim=True).clamp_min(eps)
        cos = Xn @ Yn.t()  # (M,Mp)
        return 1.0 - cos

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_bounds: List[List[Tuple[int,int]]],
        words: List[List[str]],
        labels: Optional[torch.Tensor] = None,
        return_transport: bool = False,
    ) -> SCOUTOutputs:
        device = input_ids.device
        B = input_ids.size(0)

        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B,L,D)

        C = self.num_classes
        K = self.K

        d_ck = H.new_zeros((B, C, K))
        pi = H.new_zeros((B, C, K))
        D = H.new_zeros((B, C))

        aux: Dict[str, Any] = {"per_example": []}

        b = torch.full((self.Mp,), 1.0 / self.Mp, device=device, dtype=H.dtype)

        for i in range(B):
            dist_info = self._build_span_distribution(H[i], words[i], word_bounds[i])
            spans_i: List[SpanInfo] = dist_info["spans"]
            X = dist_info["emb"]  # (M,D)
            a = dist_info["a"]    # (M,)

            if X.size(0) == 0:
                # If no spans, set large distances
                d_ck[i].fill_(1e3)
                D[i].fill_(1e3)
                pi[i].fill_(1.0 / K)
                aux["per_example"].append({"spans": [], "evidence": [], "links": [], "reg": torch.tensor(0.0, device=device, dtype=H.dtype)})
                continue

            # Precompute token intervals for reg/explanation
            span_word_ranges = [(s.word_start, s.word_end) for s in spans_i]
            span_tok_ranges = [(s.tok_start, s.tok_end) for s in spans_i]

            # Compute per class/proto distances
            T_cache: Dict[Tuple[int,int], torch.Tensor] = {}
            m_cache: Dict[Tuple[int,int], torch.Tensor] = {}

            for c in range(C):
                for k in range(K):
                    Y = self.proto_supports[c, k]  # (Mp,D)
                    cost = self._cosine_cost(X, Y)  # (M,Mp)
                    T, logT = uot_sinkhorn_log(
                        cost, a, b,
                        eps=self.uot_eps,
                        rho1=self.uot_rho, rho2=self.uot_rho,
                        n_iters=self.uot_iters,
                    )
                    obj = uot_objective(
                        T, logT, cost, a, b,
                        eps=self.uot_eps, rho1=self.uot_rho, rho2=self.uot_rho,
                    )
                    d_ck[i, c, k] = obj
                    m_cache[(c,k)] = T.sum(dim=1)  # span transported mass
                    if return_transport:
                        T_cache[(c,k)] = T

                # soft-min aggregation across prototypes
                D[i, c] = -(1.0 / self.gamma) * torch.logsumexp(-self.gamma * d_ck[i, c], dim=0)
                # responsibilities
                pi[i, c] = torch.softmax(-self.gamma * d_ck[i, c], dim=0)

            logits = -self.beta * D[i]
            probs_i = torch.softmax(logits, dim=0)


            # Regularization term (Eq. (25)) uses transported span mass for the *gold* class.
            reg_i = None
            if labels is not None and self.lam_reg > 0:
                y_true = int(labels[i].item())
                mix_m_true = torch.zeros((X.size(0),), device=device, dtype=H.dtype)
                for k in range(K):
                    mix_m_true = mix_m_true + pi[i, y_true, k].clone() * m_cache[(y_true, k)].clone()
                # Token importance projection w = P m (Eq. (23)-(24)) without explicitly materializing P
                L_active = int(attention_mask[i].sum().item())
                if L_active <= 0:
                    reg_i = mix_m_true.sum()
                else:
                    w = torch.zeros((L_active,), device=device, dtype=H.dtype)
                    for si, (ts, te) in enumerate(span_tok_ranges):
                        if ts >= L_active:
                            continue
                        te2 = min(te, L_active - 1)
                        span_len = float(te2 - ts + 1)
                        if span_len <= 0:
                            continue
                        w[ts:te2+1] += mix_m_true[si] / span_len
                    tv = torch.abs(w[1:] - w[:-1]).sum() if L_active >= 2 else torch.tensor(0.0, device=device, dtype=H.dtype)
                    reg_i = mix_m_true.sum() + self.eta_tv * tv

            # Evidence mass for explanation: mixture of true-class prototypes if labels else predicted class
            y_for_exp = int(labels[i].item()) if labels is not None else int(probs_i.argmax().item())
            mix_m = None
            mix_links = []

            if return_transport and (y_for_exp is not None):
                mix_m = torch.zeros((X.size(0),), device=device, dtype=H.dtype)
                for k in range(K):
                    mk = m_cache[(y_for_exp, k)]  # (M,)
                    mix_m = mix_m + pi[i, y_for_exp, k] * mk

                # Build top evidence spans
                top_e = min(5, mix_m.numel())
                ev_vals, ev_idx = torch.topk(mix_m, k=top_e, largest=True)
                evidence = []
                for val, idx_span in zip(ev_vals.tolist(), ev_idx.tolist()):
                    ws, we = span_word_ranges[idx_span]
                    evidence.append({
                        "mass": float(val),
                        "word_start": int(ws),
                        "word_end": int(we),
                        "span_text": build_span_text(words[i], ws, we),
                    })

                # Links: for each evidence span, show top links to supports across prototypes (weighted by pi)
                for idx_span in ev_idx.tolist():
                    links = []
                    for k in range(K):
                        T = T_cache[(y_for_exp, k)]  # (M,Mp)
                        row = T[idx_span]  # (Mp,)
                        for j in range(self.Mp):
                            links.append({
                                "prototype": int(k),
                                "support": int(j),
                                "mass": float((pi[i, y_for_exp, k] * row[j]).item()),
                            })
                    links.sort(key=lambda x: -x["mass"])
                    mix_links.append({"evidence_span": int(idx_span), "top": links[:5]})

                aux["per_example"].append({
                    "span_word_ranges": span_word_ranges,
                    "span_tok_ranges": span_tok_ranges,
                    "evidence": evidence,
                    "links": mix_links,
                    "y_for_exp": int(y_for_exp),
                    "reg": reg_i,
                })
            else:
                aux["per_example"].append({
                    "span_word_ranges": span_word_ranges,
                    "span_tok_ranges": span_tok_ranges,
                    "evidence": [],
                    "links": [],
                    "y_for_exp": int(y_for_exp),
                    "reg": reg_i,
                })

        logits = -self.beta * D  # (B,C)
        probs = torch.softmax(logits, dim=1)

        loss = None
        if labels is not None:
            ce = F.cross_entropy(logits, labels)

            # Clustering & separation losses (Eq. (21)-(22))
            idx = torch.arange(B, device=device)
            clst = D[idx, labels].mean()

            # min over incorrect classes
            D_mask = D.clone()
            D_mask[idx, labels] = 1e9
            min_other = D_mask.min(dim=1).values
            sep = F.relu(self.margin - min_other).mean()


            # Regularization (Eq. (25)) computed from transported span mass (exact).
            if self.lam_reg > 0:
                reg_vals = []
                for ex in aux["per_example"]:
                    if "reg" in ex and ex["reg"] is not None:
                        reg_vals.append(ex["reg"])
                reg = torch.stack(reg_vals).mean() if len(reg_vals) > 0 else torch.tensor(0.0, device=device, dtype=H.dtype)
            else:
                reg = torch.tensor(0.0, device=device, dtype=H.dtype)

            loss = ce + self.lam_clst * clst + self.lam_sep * sep + self.lam_reg * reg

        return SCOUTOutputs(
            logits=logits,
            probs=probs,
            D=D,
            d_ck=d_ck,
            pi=pi,
            loss=loss,
            aux=aux,
        )

    @torch.no_grad()
    def predict_proba_texts(self, texts: List[str], tokenizer, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Convenience API used by some metrics. This uses basic whitespace split for simplicity.
        """
        from .datasets import build_collate_fn, TextDataset
        if max_length is None:
            max_length = self.max_length
        dummy_labels = [0] * len(texts)
        ds = TextDataset(texts, dummy_labels)
        collate = build_collate_fn(tokenizer, max_length=max_length)
        batch = collate([ds[i] for i in range(len(ds))])
        out = self.forward(
            input_ids=batch["input_ids"].to(next(self.parameters()).device),
            attention_mask=batch["attention_mask"].to(next(self.parameters()).device),
            word_bounds=batch["word_bounds"],
            words=batch["words"],
            labels=None,
            return_transport=False,
        )
        return out.probs.detach().cpu()
