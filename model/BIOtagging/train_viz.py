#!/usr/bin/env python3
from __future__ import annotations
"""Train parser head with reviewed annotations + visualizations + metrics."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from model.BIOtagging.bio_schema import (
    BIO_TAGS, BIO_TAG_TO_ID, ID_TO_BIO_TAG,
    NormalizedQuery, tokens_to_bio_labels,
)
from model.BIOtagging.config import TrainingConfig
from model.BIOtagging.query_parser_head import QueryParserHead
from model.BIOtagging.synthetic_queries import generate_synthetic_queries

OUT = Path(__file__).resolve().parent / "outputs" / "train_result"
OUT.mkdir(parents=True, exist_ok=True)

DATA = Path(__file__).resolve().parent / "data"


class BIODataset(Dataset):
    def __init__(self, queries: list[NormalizedQuery], raw_queries: list[str]) -> None:
        self.pairs = list(zip(queries, raw_queries))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, NormalizedQuery]:
        return self.pairs[idx][1], self.pairs[idx][0]


def _load_real_annotations() -> tuple[list[NormalizedQuery], list[str]]:
    ann_path = DATA / "llm_annotations_3k_reviewed.json"
    q_path = DATA / "refcoco_queries_for_annotation.json"

    if not ann_path.exists():
        print("No reviewed annotations found, using rule-based fallback")
        queries_raw = json.loads(q_path.read_text())
        anns = []
        from reasonseg.query import parse_query
        for q in queries_raw:
            anns.append(parse_query(q))
        return anns, queries_raw

    anns = json.loads(ann_path.read_text())
    queries_raw = json.loads(q_path.read_text())
    queries_raw = queries_raw[:len(anns)]
    return anns, queries_raw


def _collate_fn(
    batch: list[tuple[str, NormalizedQuery]],
    hidden_dim: int = 768,
    max_len: int = 64,
) -> dict[str, torch.Tensor]:
    labels_batch: list[list[int]] = []
    hidden_batch: list[list[list[float]]] = []

    for raw_q, struct in batch:
        tokens = raw_q.lower().split()
        labels = tokens_to_bio_labels(tokens, struct)
        hidden = [[random.uniform(-0.1, 0.1) for _ in range(hidden_dim)] for _ in tokens]
        labels_batch.append(labels)
        hidden_batch.append(hidden)

    batch_max = min(max(len(l) for l in labels_batch), max_len)
    padded_hidden, padded_labels = [], []
    for h, l in zip(hidden_batch, labels_batch):
        hl = h[:batch_max] + [[0.0] * hidden_dim] * (batch_max - len(h[:batch_max]))
        ll = l[:batch_max] + [-100] * (batch_max - len(l[:batch_max]))
        padded_hidden.append(hl)
        padded_labels.append(ll)

    return {
        "hidden": torch.tensor(padded_hidden, dtype=torch.float32),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def _build_confusion_matrix(
    model: QueryParserHead, loader: DataLoader, device: torch.device,
) -> np.ndarray:
    n_tags = model.num_tags
    cm = np.zeros((n_tags, n_tags), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            hidden = batch["hidden"].to(device)
            labels = batch["labels"].to(device)
            logits = model(hidden)
            preds = logits.argmax(dim=-1)
            for p, l in zip(preds.view(-1).cpu(), labels.view(-1).cpu()):
                if l >= 0:
                    cm[int(l), int(p)] += 1
    return cm


def _plot_loss_curve(train_losses: list[float], val_accs: list[float], save_path: Path) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="steelblue")
    ax1.plot(epochs, train_losses, color="steelblue", linewidth=1.5, marker="o", markersize=3, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Val Accuracy", color="darkorange")
    ax2.plot(epochs, val_accs, color="darkorange", linewidth=1.5, marker="s", markersize=3, label="Val Acc")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    fig.suptitle("Training Progress", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(
    cm: np.ndarray, save_path: Path, top_k: int = 14,
) -> None:
    tag_names = [ID_TO_BIO_TAG[i] for i in range(len(BIO_TAGS))]
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")
    ax1.set_xticks(range(len(tag_names)))
    ax1.set_yticks(range(len(tag_names)))
    ax1.set_xticklabels(tag_names, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(tag_names, fontsize=8)
    ax1.set_title("Confusion Matrix (Counts)", fontsize=12)
    for i in range(len(tag_names)):
        for j in range(len(tag_names)):
            if cm[i, j] > 0:
                ax1.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=5, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(cm_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(tag_names)))
    ax2.set_yticks(range(len(tag_names)))
    ax2.set_xticklabels(tag_names, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(tag_names, fontsize=8)
    ax2.set_title("Confusion Matrix (Normalized by Row)", fontsize=12)
    for i in range(len(tag_names)):
        for j in range(len(tag_names)):
            if cm_norm[i, j] > 0.05:
                ax2.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", fontsize=5)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Parser Head Confusion Matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_tag_metrics(cm: np.ndarray, save_path: Path) -> None:
    tag_names = [ID_TO_BIO_TAG[i] for i in range(len(BIO_TAGS))]
    precision, recall, f1 = [], [], []
    for i in range(len(tag_names)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f = 2 * prec * rec / (prec + rec + 1e-8)
        precision.append(prec)
        recall.append(rec)
        f1.append(f)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(tag_names))
    w = 0.25
    ax.bar(x - w, precision, w, label="Precision", color="steelblue", alpha=0.8)
    ax.bar(x, recall, w, label="Recall", color="darkorange", alpha=0.8)
    ax.bar(x + w, f1, w, label="F1", color="forestgreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tag_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Per-Tag Precision / Recall / F1", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _compute_full_metrics(
    model: QueryParserHead, loader: DataLoader, device: torch.device,
) -> dict[str, Any]:
    cm = _build_confusion_matrix(model, loader, device)
    tag_names = [ID_TO_BIO_TAG[i] for i in range(len(BIO_TAGS))]
    total = cm.sum()
    correct = cm.trace()

    per_tag: dict[str, dict[str, float]] = {}
    for i, name in enumerate(tag_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        per_tag[name] = {
            "precision": float(tp / (tp + fp + 1e-8)),
            "recall": float(tp / (tp + fn + 1e-8)),
            "f1": float(2 * tp / (2 * tp + fp + fn + 1e-8)),
            "support": int(cm[i, :].sum()),
        }

    return {
        "overall_accuracy": float(correct / total) if total > 0 else 0.0,
        "total_tokens": int(total),
        "correct_tokens": int(correct),
        "macro_f1": float(np.mean([v["f1"] for v in per_tag.values()])),
        "per_tag_metrics": per_tag,
    }


def _format_report(metrics: dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "  Parser Head Training Report",
        "=" * 60,
        f"  Overall Accuracy: {metrics['overall_accuracy']:.4f} ({100*metrics['overall_accuracy']:.1f}%)",
        f"  Correct/Total: {metrics['correct_tokens']}/{metrics['total_tokens']}",
        f"  Macro F1: {metrics['macro_f1']:.4f}",
        "",
        f"  {'Tag':<16s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Sup':>6s}",
        f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*6}",
    ]
    for tag, m in metrics["per_tag_metrics"].items():
        lines.append(
            f"  {tag:<16s} {m['precision']:7.4f} {m['recall']:7.4f} {m['f1']:7.4f} {m['support']:6d}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> int:
    config = TrainingConfig(
        synthetic_queries_count=0,
        max_epochs=30,
        batch_size=64,
        val_split_ratio=0.0,
        hidden_dim=128,
        num_decoder_layers=2,
        nhead=4,
        dim_feedforward=256,
        warmup_steps=300,
        output_dir=OUT,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ── Load data ──
    print("\nLoading data...", flush=True)
    real_qs, real_raw = _load_real_annotations()
    syn_qs = generate_synthetic_queries(500)
    syn_raw = [
        " ".join((q.get("attributes", []) or []) + ([q["target"]] if q["target"] else [])).strip()
        or "no object"
        for q in syn_qs
    ]

    all_qs = real_qs + syn_qs
    all_raw = real_raw + syn_raw
    print(f"  Real: {len(real_qs)}, Synthetic: {len(syn_qs)}, Total: {len(all_qs)}", flush=True)

    combined = list(zip(all_qs, all_raw))
    random.Random(42).shuffle(combined)
    all_qs[:] = [p[0] for p in combined]  # type: ignore[index]
    all_raw[:] = [p[1] for p in combined]  # type: ignore[index]

    val_size = max(1, int(len(all_qs) * 0.15))
    train_qs, val_qs = all_qs[:-val_size], all_qs[-val_size:]
    train_raw, val_raw = all_raw[:-val_size], all_raw[-val_size:]

    collate = lambda batch: _collate_fn(batch, hidden_dim=config.hidden_dim)
    train_loader = DataLoader(
        BIODataset(train_qs, train_raw), batch_size=config.batch_size,
        shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        BIODataset(val_qs, val_raw), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    # ── Build model ──
    model = QueryParserHead(
        hidden_dim=config.hidden_dim,
        num_tags=config.num_tags,
        num_layers=config.num_decoder_layers,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    total_steps = config.max_epochs * len(train_loader)
    warmup_ratio = min(config.warmup_steps / total_steps if total_steps else 0.0, 0.3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=total_steps,
        pct_start=warmup_ratio,
    )

    # ── Compute class weights from training data ──
    label_counts = Counter()
    temp_loader = DataLoader(
        BIODataset(train_qs, train_raw), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate,
    )
    for batch in temp_loader:
        for label in batch["labels"].view(-1).tolist():
            if label >= 0:
                label_counts[label] += 1
    total = sum(label_counts.values())
    n_classes = model.num_tags
    class_weights = torch.ones(n_classes) * 0.1
    for c, count in label_counts.items():
        class_weights[c] = max(0.05, total / (n_classes * count))
    class_weights[0] *= 0.3
    print(f"  Class weights: O={class_weights[0]:.2f}, TGT={class_weights[BIO_TAG_TO_ID['B-TGT']]:.2f}, ATTR={class_weights[BIO_TAG_TO_ID['B-ATTR']]:.2f}, REL={class_weights[BIO_TAG_TO_ID['B-REL']]:.2f}", flush=True)
    del temp_loader

    # ── Train ──
    print(f"\nTraining {config.max_epochs} epochs...", flush=True)
    train_losses: list[float] = []
    val_accs: list[float] = []
    best_val_acc = 0.0
    best_state: dict[str, Any] = {}

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            hidden = batch["hidden"].to(device)
            labels = batch["labels"].to(device)
            logits = model(hidden)
            loss = model.compute_loss(logits, labels, class_weights=class_weights, gamma=3.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        train_losses.append(avg_loss)

        model.eval()
        total_acc = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                hidden = batch["hidden"].to(device)
                labels = batch["labels"].to(device)
                logits = model(hidden)
                total_acc += model.compute_accuracy(logits, labels)
                n_batches += 1
        val_acc = total_acc / max(n_batches, 1)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  Epoch {epoch+1:3d}/{config.max_epochs}  "
            f"loss={avg_loss:.4f}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}",
            flush=True,
        )

    # ── Save best model ──
    model.load_state_dict(best_state)
    torch.save(best_state, OUT / "parser_head_best.pt")
    print(f"\nBest val acc: {best_val_acc:.4f}, model saved", flush=True)

    # ── Visualizations ──
    print("\nGenerating visualizations...", flush=True)
    _plot_loss_curve(train_losses, val_accs, OUT / "loss_curve.png")
    print(f"  Loss curve: {OUT / 'loss_curve.png'}", flush=True)

    cm = _build_confusion_matrix(model, val_loader, device)
    _plot_confusion_matrix(cm, OUT / "confusion_matrix.png")
    print(f"  Confusion matrix: {OUT / 'confusion_matrix.png'}", flush=True)

    _plot_per_tag_metrics(cm, OUT / "per_tag_metrics.png")
    print(f"  Per-tag metrics: {OUT / 'per_tag_metrics.png'}", flush=True)

    # ── Metrics report ──
    metrics = _compute_full_metrics(model, val_loader, device)
    report = _format_report(metrics)
    print(report)
    (OUT / "metrics_report.txt").write_text(report)

    json.dump(metrics, (OUT / "metrics.json").open("w"), indent=2, ensure_ascii=False)
    print(f"\nAll outputs saved to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
