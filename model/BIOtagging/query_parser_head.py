from __future__ import annotations

import torch
import torch.nn as nn

from .bio_schema import BIO_TAG_TO_ID, ID_TO_BIO_TAG, bio_tags_to_structure
from .bio_schema import NormalizedQuery


class QueryParserHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_tags: int = 14,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_tags)

    def forward(
        self,
        beit3_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask.to(dtype=torch.bool, device=beit3_hidden.device)
        x = self.transformer(beit3_hidden, src_key_padding_mask=padding_mask)
        logits = self.classifier(x)
        return logits

    def predict_tags(
        self,
        beit3_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.forward(beit3_hidden, attention_mask)
        return logits.argmax(dim=-1)

    def decode_structure(
        self,
        tokens: list[str],
        beit3_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> NormalizedQuery:
        tag_ids = self.predict_tags(beit3_hidden, attention_mask)
        tag_ids_list = tag_ids.squeeze(0).cpu().tolist()
        tag_names = [ID_TO_BIO_TAG[tid] for tid in tag_ids_list]

        seq_len = min(len(tokens), len(tag_names))
        return bio_tags_to_structure(tokens[:seq_len], tag_names[:seq_len])

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        labels_flat = labels.view(-1)
        log_probs_flat = log_probs.view(-1, logits.size(-1))

        valid_mask = labels_flat != ignore_index
        valid_labels = labels_flat[valid_mask]
        valid_log_probs = log_probs_flat[valid_mask]

        if valid_labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        ce_loss = torch.nn.functional.nll_loss(
            valid_log_probs, valid_labels, reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** gamma

        if class_weights is not None:
            class_weights = class_weights.to(logits.device)
            sample_weights = class_weights[valid_labels]
            loss = (alpha * focal_weight * sample_weights * ce_loss).mean()
        else:
            loss = (focal_weight * ce_loss).mean()

        return loss

    @staticmethod
    def compute_accuracy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> float:
        preds = logits.argmax(dim=-1).view(-1)
        labels_flat = labels.view(-1)
        mask = labels_flat != ignore_index
        if mask.sum() == 0:
            return 1.0
        return (preds[mask] == labels_flat[mask]).float().mean().item()
