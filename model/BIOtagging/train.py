from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .bio_schema import (
    BIO_TAG_TO_ID,
    NormalizedQuery,
    tokens_to_bio_labels,
)
from .config import TrainingConfig
from .query_parser_head import QueryParserHead
from .synthetic_queries import generate_synthetic_queries


class BIOTaggingDataset(Dataset):
    def __init__(
        self,
        queries: list[NormalizedQuery],
        raw_queries: list[str],
        max_seq_len: int = 64,
    ) -> None:
        self.queries = queries
        self.raw_queries = raw_queries
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> tuple[str, NormalizedQuery]:
        return self.raw_queries[idx], self.queries[idx]


class BIOTrainingRunner:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.model: QueryParserHead | None = None
        self.device = torch.device(config.device)

    def prepare_data(self) -> tuple[BIOTaggingDataset, BIOTaggingDataset]:
        queries = generate_synthetic_queries(self.config.synthetic_queries_count)
        raw_queries = [_query_to_text(q) for q in queries]

        pairs = list(zip(queries, raw_queries))
        random.Random(42).shuffle(pairs)
        queries[:] = [p[0] for p in pairs]  # type: ignore[index]
        raw_queries[:] = [p[1] for p in pairs]  # type: ignore[index]

        val_size = max(1, int(len(queries) * self.config.val_split_ratio))
        train_queries = queries[:-val_size]
        train_raw = raw_queries[:-val_size]
        val_queries = queries[-val_size:]
        val_raw = raw_queries[-val_size:]

        train_dataset = BIOTaggingDataset(train_queries, train_raw)
        val_dataset = BIOTaggingDataset(val_queries, val_raw)
        return train_dataset, val_dataset

    def train(
        self,
        train_dataset: BIOTaggingDataset,
        val_dataset: BIOTaggingDataset,
    ) -> dict[str, Any]:
        self.model = QueryParserHead(
            hidden_dim=self.config.hidden_dim,
            num_tags=self.config.num_tags,
            num_layers=self.config.num_decoder_layers,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
        ).to(self.device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = self.config.max_epochs * len(train_loader)
        warmup_ratio = min((self.config.warmup_steps / total_steps) if total_steps else 0.0, 0.3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_ratio,
        )

        best_val_acc = 0.0
        best_state: dict[str, Any] = {}
        train_losses: list[float] = []
        val_accs: list[float] = []

        for epoch in range(self.config.max_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                hidden = batch["hidden"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(hidden)
                loss = self.model.compute_loss(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(train_loader), 1)
            train_losses.append(avg_loss)

            val_acc = self._validate(val_loader)
            val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            print(
                f"Epoch {epoch+1:3d}/{self.config.max_epochs}  "
                f"loss={avg_loss:.4f}  val_acc={val_acc:.4f}  "
                f"best={best_val_acc:.4f}"
            )

        if best_state:
            self.model.load_state_dict(best_state)
            output_dir = self.config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, output_dir / self.config.model_save_name)

        return {
            "best_val_acc": best_val_acc,
            "train_losses": train_losses,
            "val_accs": val_accs,
        }

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()  # type: ignore[union-attr]
        total_acc = 0.0
        num_batches = 0

        for batch in loader:
            hidden = batch["hidden"].to(self.device)
            labels = batch["labels"].to(self.device)
            with torch.no_grad():
                logits = self.model(hidden)  # type: ignore[union-attr]
            acc = self.model.compute_accuracy(logits, labels)  # type: ignore[union-attr]
            total_acc += acc
            num_batches += 1

        return total_acc / max(num_batches, 1)

    def _collate_fn(self, batch: Sequence[tuple[str, NormalizedQuery]]) -> dict[str, torch.Tensor]:
        labels_batch: list[list[int]] = []
        hidden_batch: list[list[list[float]]] = []

        for raw_query, query_struct in batch:
            tokens = raw_query.lower().split()
            labels = tokens_to_bio_labels(tokens, query_struct)
            hidden = [_random_embedding(self.config.hidden_dim) for _ in tokens]
            labels_batch.append(labels)
            hidden_batch.append(hidden)

        max_len = min(
            max(len(lab) for lab in labels_batch),
            64,
        )

        padded_hidden = []
        padded_labels = []
        masks = []

        for hidden, labels in zip(hidden_batch, labels_batch):
            h = hidden[:max_len] + [[0.0] * self.config.hidden_dim] * (max_len - len(hidden))
            l = labels[:max_len] + [-100] * (max_len - len(labels))
            m = [1.0] * min(len(hidden), max_len) + [0.0] * (max_len - len(hidden))
            padded_hidden.append(h)
            padded_labels.append(l)
            masks.append(m)

        return {
            "hidden": torch.tensor(padded_hidden, dtype=torch.float32),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "mask": torch.tensor(masks, dtype=torch.bool),
        }


def _random_embedding(dim: int) -> list[float]:
    return [random.uniform(-0.1, 0.1) for _ in range(dim)]


def _query_to_text(query: NormalizedQuery) -> str:
    if not query["exists"]:
        if "absent_object" in query["negatives"] and query["target"]:
            return f"no {query['target']}"
        return "nothing"

    parts: list[str] = list(query["attributes"])
    if query["target"]:
        parts.append(query["target"])
    for rel in query["relations"]:
        parts.extend([rel["type"], rel["target"]])
    for act in query["actions"]:
        parts.append(act["verb"])
        if act["target"]:
            parts.append(act["target"])
    return " ".join(parts)
