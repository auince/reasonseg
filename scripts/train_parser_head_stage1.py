#!/usr/bin/env python3
from __future__ import annotations
"""Stage 1: Train parser head on BEiT3 features (real) or random embeddings (fallback)."""

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BIO_ROOT = ROOT / "model" / "BIOtagging"

from model.BIOtagging.bio_schema import (
    BIO_TAGS, BIO_TAG_TO_ID, ID_TO_BIO_TAG,
    NormalizedQuery, structure_to_bio_tags, tokens_to_bio_labels,
)
from model.BIOtagging.query_parser_head import QueryParserHead

DEFAULT_OUT = BIO_ROOT / "outputs" / "stage1"
DEFAULT_DATA = ROOT / "dataset"
DEFAULT_LABELS = BIO_ROOT / "data" / "llm_annotations_3k_reviewed.json"
DEFAULT_QUERIES = BIO_ROOT / "data" / "refcoco_queries_for_annotation.json"
DEFAULT_CONFIG = ROOT / "configs" / "refcoco" / "refcoco_reasonseg.yaml"
N_TAGS = 14
HIDDEN_DIM = 768


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1 parser head training")
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--use-real-beit3", action="store_true",
                   help="Load real BEiT3 model (requires weights)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def load_silver_data(labels_path: Path, queries_path: Path) -> tuple[dict[str, NormalizedQuery], list[str]]:
    annotations = json.loads(labels_path.read_text())
    raw_queries = json.loads(queries_path.read_text())
    lookup: dict[str, NormalizedQuery] = {}
    for q, ann in zip(raw_queries, annotations):
        lookup[q.strip().lower()] = ann
    return lookup, raw_queries


def try_load_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
        local = str(ROOT / "models" / "evf-sam2-multitask")
        return AutoTokenizer.from_pretrained(local, use_fast=True)
    except Exception:
        return None


def expand_word_labels_to_subwords(
    word_bio: list[str], word_ids: list[int | None], seq_len: int,
) -> list[int]:
    labels = [-100] * seq_len
    if not word_bio:
        return labels
    last_wid = -1
    for i, wid in enumerate(word_ids[:seq_len]):
        if wid is None or wid >= len(word_bio):
            labels[i] = -100
        elif wid != last_wid:
            last_wid = wid
            labels[i] = BIO_TAG_TO_ID.get(word_bio[wid], 0)
        else:
            itag = word_bio[wid].replace("B-", "I-") if word_bio[wid].startswith("B-") else word_bio[wid]
            labels[i] = BIO_TAG_TO_ID.get(itag, 0)
    return labels


def make_random_hidden(seq_len: int, dim: int = HIDDEN_DIM) -> torch.Tensor:
    return torch.randn(1, seq_len, dim)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        t = targets.view(-1)
        lp = log_probs.view(-1, log_probs.size(-1))
        m = t != self.ignore_index
        vlp = lp[m]; vt = t[m]
        if vt.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        ce = F.nll_loss(vlp, vt, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def compute_class_weights(silvers: list[NormalizedQuery], raw_queries: list[str]) -> torch.Tensor:
    cnt = Counter()
    for q, s in zip(raw_queries, silvers):
        tokens = q.lower().split()
        labels = tokens_to_bio_labels(tokens, s)
        for l in labels:
            cnt[l] += 1
    total = sum(cnt.values())
    w = torch.ones(N_TAGS) * 0.1
    for c, n in cnt.items():
        if n > 0 and c < N_TAGS:
            w[c] = max(0.05, total / (N_TAGS * n))
    w[0] *= 0.3
    return w


def run_validation(model, tokenizer, silver_lookup, val_queries, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for q in val_queries:
            silver = silver_lookup.get(q.strip().lower(), {
                "target": None, "attributes": [], "relations": [],
                "actions": [], "negatives": [], "exists": False,
            })
            word_tokens = q.lower().split()
            if not word_tokens:
                continue
            word_bio = structure_to_bio_tags(silver, word_tokens)
            seq_len = min(len(word_tokens), 64)
            hidden = make_random_hidden(seq_len).to(device)
            logits = model(hidden)
            labels = tokens_to_bio_labels(word_tokens[:seq_len], silver)
            labels_t = torch.tensor(labels, device=device).unsqueeze(0)
            preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            for p, l in zip(preds, labels):
                if l >= 0:
                    total += 1
                    if p == l:
                        correct += 1
    return correct / max(total, 1)


def plot_results(train_losses, val_accs, val_every, output_dir):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color="steelblue")
    ax1.plot(train_losses, color="steelblue", alpha=0.4, linewidth=0.5, label="Loss")
    if len(train_losses) > 30:
        w = min(50, len(train_losses) // 4)
        sma = np.convolve(train_losses, np.ones(w) / w, mode="valid")
        ax1.plot(range(w - 1, len(train_losses)), sma, color="steelblue", linewidth=1.5, label=f"SMA({w})")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Val Accuracy", color="darkorange")
    xv = [(i + 1) * val_every for i in range(len(val_accs))]
    ax2.plot(xv, val_accs, color="darkorange", marker="o", ms=3, linewidth=1.5, label="Val Acc")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.suptitle("Stage 1: Parser Head Training")
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}", flush=True)

    silver_lookup, all_queries = load_silver_data(DEFAULT_LABELS, DEFAULT_QUERIES)
    print(f"Silver labels: {len(silver_lookup)} entries", flush=True)

    tokenizer = try_load_tokenizer()
    use_real = args.use_real_beit3 and tokenizer is not None
    print(f"Tokenizer: {'XLM-RoBERTa (subword)' if tokenizer else 'whitespace (fallback)'}", flush=True)
    print(f"Backend: {'BEiT3 (real)' if use_real else 'random embeddings'}", flush=True)

    if use_real:
        print("Loading BEiT3 model...", flush=True)
        from reasonseg.modeling.open_world_sam2_config import add_open_world_sam2_config
        from detectron2.config import get_cfg
        cfg = get_cfg()
        add_open_world_sam2_config(cfg)
        cfg.MODEL.OpenWorldSAM2.LEARNED_PARSER.ENABLED = True
        cfg.MODEL.OpenWorldSAM2.HF_LOCAL_FILES_ONLY = True
        cfg.MODEL.OpenWorldSAM2.LOCAL_TOKENIZER_CONFIG = str(ROOT / "models" / "evf-sam2-multitask")
        cfg.MODEL.OpenWorldSAM2.LOCAL_EVF_CONFIG = str(ROOT / "models" / "evf-sam2-multitask")
        from reasonseg.modeling.open_world_sam2 import OpenWorldSAM2
        model = OpenWorldSAM2.from_config(cfg)
        for n, p in model.named_parameters():
            p.requires_grad = False
        model.parser_head.train()
        for p in model.parser_head.parameters():
            p.requires_grad = True
        model.to(device)
        print(f"Model loaded, parser_head params: {sum(p.numel() for p in model.parser_head.parameters()):,}", flush=True)
    else:
        model = QueryParserHead(
            hidden_dim=HIDDEN_DIM, num_tags=N_TAGS,
            num_layers=2, nhead=8, dim_feedforward=1024, dropout=0.1,
        ).to(device)
        print(f"Parser head params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    class_weights = compute_class_weights(
        [silver_lookup[q.strip().lower()] for q in all_queries[:3000]], all_queries[:3000]
    )
    print(f"Class weights: O={class_weights[0]:.2f} TGT={class_weights[1]:.2f} ATTR={class_weights[3]:.2f}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = FocalLoss(gamma=2.0)

    train_losses: list[float] = []
    val_accs: list[float] = []
    best_val_acc = 0.0
    val_queries = random.Random(42).sample(all_queries, min(300, len(all_queries)))

    print(f"\nTraining {args.max_iters} iters (batch={args.batch_size})...", flush=True)
    pbar = tqdm(range(args.max_iters), desc="Stage 1")

    for iter_idx in pbar:
        bq = random.Random(iter_idx).sample(all_queries, args.batch_size)
        bsilver = [silver_lookup.get(q.strip().lower(), {
            "target": None, "attributes": [], "relations": [],
            "actions": [], "negatives": [], "exists": False,
        }) for q in bq]

        if use_real and tokenizer:
            enc = tokenizer(bq, return_tensors="pt", padding=True, truncation=True, max_length=64)
            ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
            model.eval(); model.parser_head.train()
            with torch.no_grad():
                out = model.mm_extractor.beit3(visual_tokens=None, textual_tokens=ids, text_padding_position=~attn)
            hidden = out["encoder_out"]
            logits = model.parser_head(hidden)
            all_labels: list[list[int]] = []
            for q, s in zip(bq, bsilver):
                wt = q.lower().split()
                wb = structure_to_bio_tags(s, wt)
                e = tokenizer(q, return_tensors="pt", truncation=True, max_length=64)
                all_labels.append(expand_word_labels_to_subwords(wb, e.word_ids(), logits.size(1)))
            labels_t = torch.tensor(all_labels, dtype=torch.long, device=device)
        else:
            max_len = max(min(len(q.split()), 64) for q in bq)
            hidden_batch, labels_batch = [], []
            for q, s in zip(bq, bsilver):
                tokens = q.lower().split()[:max_len]
                labels_batch.append(tokens_to_bio_labels(tokens, s)[:max_len] + [-100] * (max_len - len(tokens)))
                hidden_batch.append(torch.randn(1, max_len + 4, HIDDEN_DIM).squeeze(0)[:max_len])
            hidden = torch.stack(hidden_batch).to(device)
            labels_t = torch.tensor(labels_batch, dtype=torch.long, device=device)
            logits = model(hidden)

        loss = criterion(logits, labels_t)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

        if (iter_idx + 1) % args.val_every == 0:
            val_acc = run_validation(model, tokenizer, silver_lookup, val_queries[:100], device)
            val_accs.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.output_dir / "parser_head_best.pt")
            model.train()
            avg_l = sum(train_losses[-args.val_every:]) / args.val_every
            pbar.set_postfix(loss=f"{avg_l:.3f}", val=f"{val_acc:.3f}", best=f"{best_val_acc:.3f}")

    if best_val_acc == 0:
        best_val_acc = run_validation(model, tokenizer, silver_lookup, val_queries[:100], device)
        torch.save(model.state_dict(), args.output_dir / "parser_head_best.pt")

    plot_results(train_losses, val_accs, args.val_every, args.output_dir)
    print(f"\nBest val acc: {best_val_acc:.4f}", flush=True)
    print(f"Model: {args.output_dir / 'parser_head_best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
