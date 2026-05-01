#!/usr/bin/env python3
from __future__ import annotations
"""Stage 1 FAST: Pre-compute BEiT3 hidden states once, then train parser head on cached features."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BIO_ROOT = ROOT / "model" / "BIOtagging"

from model.BIOtagging.bio_schema import (
    BIO_TAGS, BIO_TAG_TO_ID, ID_TO_BIO_TAG,
    NormalizedQuery, structure_to_bio_tags, tokens_to_bio_labels,
)
from model.BIOtagging.query_parser_head import QueryParserHead

DEFAULT_OUT = BIO_ROOT / "outputs" / "stage1_fast"
DEFAULT_LABELS = BIO_ROOT / "data" / "llm_annotations_3k_reviewed.json"
DEFAULT_QUERIES = BIO_ROOT / "data" / "refcoco_queries_for_annotation.json"
DEFAULT_EXPANDED_SILVER = BIO_ROOT / "data" / "expanded_silver.json"
BEIT3_HIDDEN_DIM = 1024
N_TAGS = 14


class _BEiT3InferenceAdapter(nn.Module):
    def __init__(self, beit3_wrapper: nn.Module) -> None:
        super().__init__()
        self.beit3_wrapper = beit3_wrapper

    def forward(self, textual_tokens: torch.Tensor, text_padding_position: torch.Tensor) -> torch.Tensor:
        out = self.beit3_wrapper.beit3(
            visual_tokens=None,
            textual_tokens=textual_tokens,
            text_padding_position=text_padding_position,
        )
        return out["encoder_out"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1 Fast: cached BEiT3 + parser head training")
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="GPU index or comma-separated GPU indices, e.g. '0' or '0,1'. Use 'cpu' for CPU.",
    )
    p.add_argument("--silver-path", type=Path, default=None,
                   help="Path to expanded silver JSON [[query, structure], ...]. "
                         "When provided, overrides default 3k annotation data.")
    return p


def count_non_o_labels(query: str, structure: NormalizedQuery) -> int:
    return sum(1 for label in tokens_to_bio_labels(query.lower().split(), structure) if label != 0)


def select_silver_structure(query: str, llm_structure: NormalizedQuery | None) -> NormalizedQuery:
    from reasonseg.query import parse_query

    rule_structure = parse_query(query)
    if llm_structure is None:
        return rule_structure
    llm_score = count_non_o_labels(query, llm_structure)
    rule_score = count_non_o_labels(query, rule_structure)
    if llm_score >= rule_score and llm_score > 0:
        return llm_structure
    return rule_structure


def load_silver_data(labels_path: Path, queries_path: Path) -> tuple[list[NormalizedQuery], list[str]]:
    annotations = json.loads(labels_path.read_text())
    raw_queries = json.loads(queries_path.read_text())
    selected_annotations: list[NormalizedQuery] = []
    selected_queries: list[str] = []
    for q, ann in zip(raw_queries, annotations):
        selected_queries.append(q)
        selected_annotations.append(select_silver_structure(q, ann))
    return selected_annotations, selected_queries


def load_expanded_silver(path: Path) -> tuple[list[NormalizedQuery], list[str]]:
    data = json.loads(path.read_text())
    queries: list[str] = []
    structures: list[NormalizedQuery] = []
    for item in data:
        queries.append(item[0])
        structures.append(item[1])
    return structures, queries


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
    return beit3, tokenizer


def parse_device_spec(spec: str) -> torch.device | list[int]:
    if spec == "cpu":
        return torch.device("cpu")
    if "," in spec:
        return [int(part.strip()) for part in spec.split(",") if part.strip()]
    if spec.isdigit():
        return torch.device(f"cuda:{spec}")
    return torch.device(spec)


def get_primary_device(device: torch.device | list[int]) -> torch.device:
    if isinstance(device, list):
        return torch.device(f"cuda:{device[0]}")
    return device


def precompute_hidden_states(beit3, tokenizer, queries, device, max_len=24, precompute_bs=64):
    """Run BEiT3 once on all queries, save [seq_len, 1024] per query."""
    primary_device = get_primary_device(device)
    model: nn.Module
    if isinstance(device, list) and len(device) >= 2:
        model = nn.DataParallel(_BEiT3InferenceAdapter(beit3), device_ids=device)
    else:
        model = _BEiT3InferenceAdapter(beit3)
    model.eval()
    model.to(primary_device)
    hidden_list = []
    print(f"Precomputing BEiT3 hidden states for {len(queries)} queries...", flush=True)
    for start in tqdm(range(0, len(queries), precompute_bs), desc="BEiT3"):
        batch_queries = queries[start:start + precompute_bs]
        enc = tokenizer(
            batch_queries,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            padding=True,
        )
        ids = enc["input_ids"].to(primary_device)
        attn = enc["attention_mask"].to(primary_device)
        with torch.no_grad():
            batch_hidden = model(ids, attn == 0).cpu()
        for i in range(len(batch_queries)):
            valid_len = int(attn[i].sum().item())
            hidden_list.append(batch_hidden[i, :valid_len])
    return hidden_list


def build_labels(queries, silvers, tokenizer, max_len=24):
    labels_list = []
    for q, s in tqdm(zip(queries, silvers), desc="Labels", total=len(queries)):
        wt = q.lower().split()
        wb = structure_to_bio_tags(s, wt)
        enc = tokenizer(q, return_tensors="pt", max_length=max_len, truncation=True)
        wids = enc.word_ids()
        labels = [-100] * max_len
        last_wid = -1
        for i, wid in enumerate(wids[:max_len]):
            if wid is None or wid >= len(wb):
                labels[i] = -100
            elif wid != last_wid:
                last_wid = wid
                labels[i] = BIO_TAG_TO_ID.get(wb[wid], 0)
            else:
                itag = wb[wid].replace("B-", "I-") if wb[wid].startswith("B-") else wb[wid]
                labels[i] = BIO_TAG_TO_ID.get(itag, 0)
        labels_list.append(torch.tensor(labels, dtype=torch.long))
    return labels_list


class SequenceListDataset(Dataset):
    def __init__(self, hidden_list, labels_list):
        self.hidden_list = hidden_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.hidden_list)

    def __getitem__(self, idx):
        return self.hidden_list[idx], self.labels_list[idx]


def build_length_bucketed_loader(hidden_list, labels_list, batch_size, shuffle):
    trimmed_labels = [labels[: hidden.size(0)] for hidden, labels in zip(hidden_list, labels_list)]
    dataset = SequenceListDataset(hidden_list, trimmed_labels)
    buckets = defaultdict(list)
    for idx, hidden in enumerate(hidden_list):
        buckets[int(hidden.size(0))].append(idx)

    rng = random.Random() if shuffle else random.Random(42)
    batches = []
    for indices in buckets.values():
        ordered = list(indices)
        if shuffle:
            rng.shuffle(ordered)
        for start in range(0, len(ordered), batch_size):
            batches.append(ordered[start:start + batch_size])
    if shuffle:
        rng.shuffle(batches)

    return DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
        ),
    )


def main() -> int:
    args = build_parser().parse_args()
    device = parse_device_spec(args.device)
    primary_device = get_primary_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_silvers: list[NormalizedQuery]
    all_queries: list[str]
    if args.silver_path:
        all_silvers, all_queries = load_expanded_silver(args.silver_path)
        print(f"Expanded silver: {len(all_silvers)} entries", flush=True)
    else:
        all_silvers, all_queries = load_silver_data(DEFAULT_LABELS, DEFAULT_QUERIES)
        print(f"Silver labels: {len(all_silvers)} entries", flush=True)

    pairs = list(zip(all_queries, all_silvers))
    random.Random(42).shuffle(pairs)
    all_queries = [query for query, _ in pairs]
    all_silvers = [silver for _, silver in pairs]

    val_size = min(300, max(1, len(all_queries) // 10))
    train_queries = all_queries[:-val_size]
    train_silvers = all_silvers[:-val_size]
    val_queries = all_queries[-val_size:]
    val_silvers = all_silvers[-val_size:]

    beit3, tokenizer = load_beit3_and_tokenizer()
    max_len = 24

    # Phase 1: Precompute all hidden states ONCE
    train_hidden = precompute_hidden_states(beit3, tokenizer, train_queries, device, max_len)
    val_hidden = precompute_hidden_states(beit3, tokenizer, val_queries, device, max_len)

    train_labels = build_labels(train_queries, train_silvers, tokenizer, max_len)
    val_labels = build_labels(val_queries, val_silvers, tokenizer, max_len)

    print(f"Train: {len(train_hidden)} samples, Val: {len(val_hidden)} samples", flush=True)

    # Phase 2: Train parser head (fast, large batch)
    parser_head = QueryParserHead(
        hidden_dim=BEIT3_HIDDEN_DIM, num_tags=N_TAGS,
        num_layers=1, nhead=8, dim_feedforward=1024, dropout=0.1,
    )
    if isinstance(device, list) and len(device) >= 2:
        parser_head = nn.DataParallel(parser_head, device_ids=device)
        parser_head.to(primary_device)
    else:
        parser_head = parser_head.to(primary_device)
    print(f"Parser head params: {sum(p.numel() for p in parser_head.parameters()):,}", flush=True)

    # Class weights
    cnt = Counter()
    for tq_, ts_ in zip(train_queries, train_silvers):
        for l in tokens_to_bio_labels(tq_.lower().split(), ts_):
            if l < N_TAGS: cnt[l] += 1
    total = sum(cnt.values())
    class_weights = torch.ones(N_TAGS) * 0.1
    for c, n in cnt.items():
        if n > 0:
            class_weights[c] = min(5.0, max(0.05, total / (N_TAGS * n)))
    class_weights[0] = max(0.05, class_weights[0] * 0.5)
    class_weights = class_weights.to(primary_device)

    train_loader = build_length_bucketed_loader(train_hidden, train_labels, args.batch_size, shuffle=True)
    val_loader = build_length_bucketed_loader(val_hidden, val_labels, args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(parser_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs)
    loss_model = parser_head.module if isinstance(parser_head, nn.DataParallel) else parser_head

    train_losses: list[float] = []
    val_accs: list[float] = []
    best_val_acc = 0.0

    print(f"\nTraining {args.max_epochs} epochs (batch={args.batch_size})...", flush=True)
    for epoch in range(args.max_epochs):
        parser_head.train()
        epoch_loss = 0.0
        for h, l in train_loader:
            h, l = h.to(primary_device), l.to(primary_device)
            logits = parser_head(h)
            loss = loss_model.compute_loss(logits, l, class_weights=class_weights, gamma=2.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parser_head.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        train_losses.append(avg_loss)
        scheduler.step()

        parser_head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for h, l in val_loader:
                h, l = h.to(primary_device), l.to(primary_device)
                preds = parser_head(h).argmax(dim=-1)
                mask = l >= 0
                correct += (preds[mask] == l[mask]).sum().item()
                total += mask.sum().item()
        val_acc = correct / max(total, 1)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state_dict = parser_head.module.state_dict() if isinstance(parser_head, nn.DataParallel) else parser_head.state_dict()
            torch.save(state_dict, args.output_dir / "parser_head_best.pt")

        print(f"Epoch {epoch+1:3d}/{args.max_epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f} best={best_val_acc:.4f}", flush=True)

    _plot(train_losses, val_accs, args.output_dir)
    _compute_per_tag(parser_head, val_hidden, val_labels, primary_device, args, args.output_dir)
    print(f"\nBest: {best_val_acc:.4f} -> {args.output_dir/'parser_head_best.pt'}", flush=True)
    return 0


def _plot(losses, accs, out_dir):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="steelblue")
    ax1.plot(losses, color="steelblue", lw=1.5, marker="o", ms=3)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Val Accuracy", color="darkorange")
    ax2.plot(accs, color="darkorange", lw=1.5, marker="s", ms=3)
    ax2.tick_params(axis="y", labelcolor="darkorange")
    fig.suptitle("Stage 1: Parser Head (Pretrained BEiT3)")
    fig.tight_layout(); fig.savefig(out_dir/"stage1_loss.png", dpi=150)
    plt.close(fig)


def _compute_per_tag(model, val_hidden, val_labels, device, args, out_dir):
    cm = torch.zeros(N_TAGS, N_TAGS, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for h, l in build_length_bucketed_loader(val_hidden, val_labels, args.batch_size, shuffle=False):
            h = h.to(device)
            preds = model(h).argmax(dim=-1).cpu()
            l = l.cpu()
            for p, gt in zip(preds.view(-1), l.view(-1)):
                if gt >= 0:
                    cm[gt, p] += 1
    lines = ["Per-Tag Metrics (Pretrained BEiT3):", f"{'Tag':<16s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Sup':>6s}"]
    for i in range(N_TAGS):
        tp = cm[i,i].item(); fp = cm[:,i].sum().item()-tp; fn = cm[i,:].sum().item()-tp
        p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8); f = 2*p*r/(p+r+1e-8); s = int(cm[i,:].sum())
        if s > 0:
            lines.append(f"{ID_TO_BIO_TAG[i]:<16s} {p:7.4f} {r:7.4f} {f:7.4f} {s:6d}")
    report = "\n".join(lines)
    print(report)
    (out_dir/"metrics.txt").write_text(report)


if __name__ == "__main__":
    raise SystemExit(main())
