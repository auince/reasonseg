#!/usr/bin/env python3
from __future__ import annotations
"""Stage 1 REAL: Train parser head on actual BEiT3 hidden states (torchscale, random init)."""

import argparse
import json
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

DEFAULT_OUT = BIO_ROOT / "outputs" / "stage1_real_beit3"
DEFAULT_LABELS = BIO_ROOT / "data" / "llm_annotations_3k_reviewed.json"
DEFAULT_QUERIES = BIO_ROOT / "data" / "refcoco_queries_for_annotation.json"
BEIT3_HIDDEN_DIM = 1024
N_TAGS = 14


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1: Real BEiT3 parser head training")
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    return p


def load_silver_data(labels_path: Path, queries_path: Path) -> tuple[dict[str, NormalizedQuery], list[str]]:
    annotations = json.loads(labels_path.read_text())
    raw_queries = json.loads(queries_path.read_text())
    lookup: dict[str, NormalizedQuery] = {}
    for q, ann in zip(raw_queries, annotations):
        lookup[q.strip().lower()] = ann
    return lookup, raw_queries


def load_beit3_and_tokenizer():
    print("Loading BEiT3 with pretrained weights...", flush=True)
    from reasonseg.backends._bootstrap import ensure_root_model_package_loaded
    ensure_root_model_package_loaded()
    from reasonseg.backends.beit3.modeling_utils import _get_large_config, BEiT3Wrapper
    cfg = _get_large_config()
    beit3 = BEiT3Wrapper(cfg)

    ckpt_path = ROOT / "checkpoints" / "beit3_large_patch16_224.pth"
    if ckpt_path.is_file():
        state = torch.load(str(ckpt_path), map_location="cpu")
        from reasonseg.modeling.evf_sam2 import _load_state_dict_or_raise, _BEIT_PRETRAIN_UNEXPECTED_KEYS
        _load_state_dict_or_raise(
            beit3, state["model"],
            context="BEiT3 pretrained",
            allowed_unexpected_keys=_BEIT_PRETRAIN_UNEXPECTED_KEYS,
        )
        print(f"Loaded pretrained BEiT3 from {ckpt_path}", flush=True)
    else:
        print(f"WARNING: {ckpt_path} not found, using random init", flush=True)

    print(f"BEiT3 params: {sum(p.numel() for p in beit3.parameters()):,}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "models" / "evf-sam2-multitask"), use_fast=True,
    )
    print(f"Tokenizer: {tokenizer.__class__.__name__}", flush=True)
    return beit3, tokenizer


def compute_class_weights(queries: list[str], silvers: list[NormalizedQuery]) -> torch.Tensor:
    cnt = Counter()
    for q, s in zip(queries, silvers):
        labels = tokens_to_bio_labels(q.lower().split(), s)
        for l in labels:
            cnt[l] += 1
    total = sum(cnt.values())
    w = torch.ones(N_TAGS) * 0.1
    for c, n in cnt.items():
        if n > 0 and c < N_TAGS:
            w[c] = max(0.05, total / (N_TAGS * n))
    w[0] *= 0.3
    return w


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        lp = F.log_softmax(logits, dim=-1)
        t = targets.view(-1)
        lpf = lp.view(-1, lp.size(-1))
        m = t != self.ignore_index
        vlp, vt = lpf[m], t[m]
        if vt.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        ce = F.nll_loss(vlp, vt, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


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


def run_validation(parser_head, beit3, tokenizer, silver_lookup, val_qs, device):
    parser_head.eval()
    beit3.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for q in val_qs:
            silver = silver_lookup.get(q.strip().lower(), {
                "target": None, "attributes": [], "relations": [],
                "actions": [], "negatives": [], "exists": False,
            })
            wt = q.lower().split()
            if not wt:
                continue
            wb = structure_to_bio_tags(silver, wt)
            try:
                enc = tokenizer(q, return_tensors="pt", max_length=64, truncation=True)
                ids = enc["input_ids"].to(device)
                attn = enc["attention_mask"].to(device)
                out = beit3.beit3(visual_tokens=None, textual_tokens=ids, text_padding_position=~attn)
                logits = parser_head(out["encoder_out"])
                wids = enc.word_ids()
                labels = expand_word_labels_to_subwords(wb, wids, logits.size(1))
                preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
                for p, l in zip(preds, labels):
                    if l >= 0:
                        total += 1
                        if p == l:
                            correct += 1
            except Exception:
                continue
    return correct / max(total, 1)


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    silver_lookup, all_queries = load_silver_data(DEFAULT_LABELS, DEFAULT_QUERIES)
    print(f"Silver labels: {len(silver_lookup)} entries", flush=True)

    beit3, tokenizer = load_beit3_and_tokenizer()
    beit3 = beit3.to(device)
    for p in beit3.parameters():
        p.requires_grad = False
    beit3.eval()

    parser_head = QueryParserHead(
        hidden_dim=BEIT3_HIDDEN_DIM, num_tags=N_TAGS,
        num_layers=2, nhead=8, dim_feedforward=2048, dropout=0.1,
    ).to(device)
    parser_head.train()
    print(f"Parser head params: {sum(p.numel() for p in parser_head.parameters()):,}", flush=True)

    train_queries = all_queries[:2800]
    train_silvers = [silver_lookup[q.strip().lower()] for q in train_queries]
    class_weights = compute_class_weights(train_queries, train_silvers).to(device)
    print(f"Class weights: O={class_weights[0]:.2f} TGT={class_weights[1]:.2f} ATTR={class_weights[3]:.2f}", flush=True)

    val_queries = random.Random(42).sample(all_queries, min(200, len(all_queries)))
    optimizer = torch.optim.AdamW(parser_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_iters)
    criterion = FocalLoss(gamma=2.0)

    train_losses: list[float] = []
    val_accs: list[float] = []
    best_val_acc = 0.0
    bs = args.batch_size

    print(f"\nTraining {args.max_iters} iters (batch={bs})...", flush=True)
    pbar = tqdm(range(args.max_iters), desc="Stage1-BEiT3")

    for it in pbar:
        bq = random.Random(it).sample(train_queries, bs)
        bsilver = [silver_lookup[q.strip().lower()] for q in bq]

        enc = tokenizer(bq, return_tensors="pt", padding=True, max_length=64, truncation=True)
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = beit3.beit3(visual_tokens=None, textual_tokens=ids, text_padding_position=~attn)

        encoder_out = out["encoder_out"]

        bs_actual = encoder_out.size(0)
        max_len = encoder_out.size(1)
        all_labels: list[list[int]] = []
        for qi, (q, s) in enumerate(zip(bq, bsilver)):
            wt = q.lower().split()
            wb = structure_to_bio_tags(s, wt)
            e = tokenizer(q, return_tensors="pt", max_length=64, truncation=True)
            wids = e.word_ids()
            labels = expand_word_labels_to_subwords(wb, wids, max_len)
            valid = sum(1 for l in labels if l >= 0)
            if valid < 2:
                labels = [-100] * max_len
            all_labels.append(labels)
        labels_t = torch.tensor(all_labels, dtype=torch.long, device=device)

        total_valid = (labels_t >= 0).sum().item()
        if total_valid < 4:
            continue

        logits = parser_head(encoder_out)

        loss = criterion(logits, labels_t)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parser_head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())

        if (it + 1) % args.val_every == 0:
            val_acc = run_validation(parser_head, beit3, tokenizer, silver_lookup, val_queries[:100], device)
            val_accs.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(parser_head.state_dict(), args.output_dir / "parser_head_best.pt")
            parser_head.train()
            avg_l = sum(train_losses[-args.val_every:]) / args.val_every
            pbar.set_postfix(loss=f"{avg_l:.3f}", val=f"{val_acc:.3f}", best=f"{best_val_acc:.3f}")

    if best_val_acc == 0:
        best_val_acc = run_validation(parser_head, beit3, tokenizer, silver_lookup, val_queries[:100], device)
        torch.save(parser_head.state_dict(), args.output_dir / "parser_head_best.pt")

    _plot(train_losses, val_accs, args.val_every, args.output_dir)
    print(f"\nBest val acc: {best_val_acc:.4f}", flush=True)
    print(f"Model: {args.output_dir / 'parser_head_best.pt'}", flush=True)
    return 0


def _plot(losses, val_accs, val_every, out_dir):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Loss", color="steelblue")
    ax1.plot(losses, color="steelblue", alpha=0.4, linewidth=0.5, label="Loss")
    if len(losses) > 30:
        w = min(50, len(losses)//4)
        ax1.plot(range(w-1, len(losses)), np.convolve(losses, np.ones(w)/w, mode="valid"), color="steelblue", lw=1.5)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Val Accuracy", color="darkorange")
    ax2.plot([(i+1)*val_every for i in range(len(val_accs))], val_accs, color="darkorange", marker="o", ms=3, lw=1.5)
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.suptitle("Stage 1: Parser Head Training (Real BEiT3, random init)")
    fig.tight_layout()
    fig.savefig(out_dir / "stage1_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
