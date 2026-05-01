from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .BIOtagging.bio_schema import NormalizedQuery
from .BIOtagging.query_parser_head import QueryParserHead
from .gnn import QueryGraphGAT
from .vr_ov_types import QueryGraph
from reasonseg.query import parse_query

logger = logging.getLogger(__name__)

# ── Edge index: fully-connected 4-node DAG (category→attr,rel,act;  ──────────
#    attr→rel,act; rel→act) — no self-loops, per spec.                         ─
_FULL_EDGES_4 = torch.tensor([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]], dtype=torch.long)

# ── Node-type constants ──────────────────────────────────────────────────────
_NODE_CAT = 0
_NODE_ATTR = 1
_NODE_REL = 2
_NODE_ACT = 3

_NODE_TYPE_NAMES: list[str] = ["category", "attribute", "relation", "action"]
_SPECIAL_TOKENS = {"[cls]", "[sep]", "[pad]"}


class BIOQueryParser(nn.Module):
    """Query parser powered by a trained BIO tagger.

    Pipeline
    --------
    1.  BEiT3 hidden → QueryParserHead → BIO tags → NormalizedQuery
    2.  NormalizedQuery → 4 graph nodes (category, attribute, relation, action)
    3.  GNN graph reasoning → QueryGraph

    Parameters
    ----------
    parser_checkpoint : str | None
        Path to a ``parser_head_best.pt`` checkpoint. When *None* (or the
        path does not exist) the parser intentionally falls back to the
        deterministic rule parser from ``reasonseg.query``.
    hidden_dim : int
        BEiT3 hidden-state dimensionality (default 768).
    num_tags : int
        Number of BIO tag classes (default 14).
    num_layers : int
        Transformer-encoder layers inside QueryParserHead.
    nhead : int
        Transformer-encoder attention heads inside QueryParserHead.
    gnn_hidden : int
        Hidden size for the 2-layer GAT (default 256).
    gnn_out : int
        Output feature size for the GAT (default 128).
    num_relation_types : int
        Size of the relation-type embedding vocabulary (default 50).
    num_action_types : int
        Size of the action-verb embedding vocabulary (default 50).
    num_node_types : int
        Number of semantic node types (fixed at 4).
    """

    def __init__(
        self,
        parser_checkpoint: Optional[str] = None,
        hidden_dim: int = 768,
        num_tags: int = 14,
        num_layers: int = 2,
        nhead: int = 8,
        gnn_hidden: int = 256,
        gnn_out: int = 128,
        num_relation_types: int = 50,
        num_action_types: int = 50,
        num_node_types: int = 4,
    ) -> None:
        super().__init__()

        ckpt_path = Path(parser_checkpoint) if parser_checkpoint else None
        self.parser_checkpoint = ckpt_path
        self._has_parser = ckpt_path is not None and ckpt_path.exists()
        self.parser_mode = "bio_checkpoint" if self._has_parser else "rule_fallback"
        effective_hidden_dim = hidden_dim

        if self._has_parser:
            assert ckpt_path is not None
            ckpt = torch.load(ckpt_path, map_location="cpu")
            effective_hidden_dim = ckpt["classifier.weight"].shape[1]
            ckpt_num_layers = len(
                {k.split(".")[2] for k in ckpt if k.startswith("transformer.layers")}
            )
            self.parser_head = QueryParserHead(
                hidden_dim=effective_hidden_dim,
                num_tags=num_tags,
                num_layers=ckpt_num_layers,
                nhead=nhead if effective_hidden_dim % nhead == 0 else 8,
            )
            self.parser_head.load_state_dict(ckpt)
            logger.info(
                "BIOQueryParser: loaded BIO tagger (dim=%d, layers=%d) from %s",
                effective_hidden_dim,
                ckpt_num_layers,
                ckpt_path,
            )
        else:
            self.parser_head = QueryParserHead(
                hidden_dim=effective_hidden_dim,
                num_tags=num_tags,
                num_layers=num_layers,
                nhead=nhead,
            )
            if ckpt_path is None:
                logger.warning(
                    "BIOQueryParser: no parser checkpoint configured; using deterministic rule fallback"
                )
            else:
                logger.warning(
                    "BIOQueryParser: parser checkpoint %s not found; using deterministic rule fallback",
                    ckpt_path,
                )
        self.parser_head.eval()

        self.hidden_dim = effective_hidden_dim

        # ── 2.  Embedding layers ─────────────────────────────────────────
        self.relation_embed = nn.Embedding(num_relation_types, effective_hidden_dim)
        self.action_embed = nn.Embedding(num_action_types, effective_hidden_dim)
        self.node_type_embed = nn.Embedding(num_node_types, effective_hidden_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(effective_hidden_dim, gnn_hidden),
            nn.LayerNorm(gnn_hidden),
        )

        # ── 3.  GNN ──────────────────────────────────────────────────────
        self.gnn = QueryGraphGAT(
            in_dim=gnn_hidden,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  Forward
    # ══════════════════════════════════════════════════════════════════════════

    def forward(
        self,
        beit3_hidden: Tensor,
        attention_mask: Optional[Tensor],
        tokens_list: list[str],
        query_struct: NormalizedQuery | None = None,
    ) -> QueryGraph:
        """Run the full parse pipeline (batch = 1).

        Parameters
        ----------
        beit3_hidden : Tensor
            BEiT3 encoder output, shape ``[1, seq, hidden_dim]``.
        attention_mask : Tensor | None
            Attention mask, shape ``[1, seq]``.
        tokens_list : list[str]
            Token strings corresponding to the sequence positions.

        Returns
        -------
        QueryGraph
            Four-node graph with per-node GNN-enhanced features.
        """
        self._validate_encoder_inputs(
            beit3_hidden=beit3_hidden,
            attention_mask=attention_mask,
            tokens_list=tokens_list,
        )
        normalized_query = query_struct or self.decode_to_structure(
            beit3_hidden=beit3_hidden,
            attention_mask=attention_mask,
            tokens_list=tokens_list,
        )
        return self.encode_structure(
            normalized_query,
            beit3_hidden=beit3_hidden,
            tokens_list=tokens_list,
        )

    def encode_structure(
        self,
        query_struct: NormalizedQuery,
        *,
        beit3_hidden: Tensor,
        tokens_list: list[str],
    ) -> QueryGraph:
        """Build a ``QueryGraph`` directly from canonical structured input."""
        self._validate_encoder_inputs(
            beit3_hidden=beit3_hidden,
            attention_mask=None,
            tokens_list=tokens_list,
        )
        device = beit3_hidden.device
        nodes, node_types = self._build_graph_nodes(
            query_struct,
            beit3_hidden,
            tokens_list,
            device,
        )
        nodes_proj = self.node_proj(nodes)
        edges = _FULL_EDGES_4.to(device)
        enhanced = self.gnn(nodes_proj, edges)
        return QueryGraph(
            nodes=[enhanced[index] for index in range(4)],
            edges=edges,
            node_types=node_types,
        )

    def decode_to_structure(
        self,
        beit3_hidden: Tensor,
        attention_mask: Optional[Tensor],
        tokens_list: list[str],
    ) -> NormalizedQuery:
        """Decode tokens into the canonical normalized query structure."""
        self._validate_encoder_inputs(
            beit3_hidden=beit3_hidden,
            attention_mask=attention_mask,
            tokens_list=tokens_list,
        )
        if self._has_parser:
            return self.parser_head.decode_structure(
                tokens_list, beit3_hidden, attention_mask
            )
        return self._structure_fallback(tokens_list)

    def _validate_encoder_inputs(
        self,
        *,
        beit3_hidden: Tensor,
        attention_mask: Optional[Tensor],
        tokens_list: list[str],
    ) -> None:
        if beit3_hidden.ndim != 3 or beit3_hidden.shape[0] != 1:
            raise ValueError(
                f"BIOQueryParser expects beit3_hidden with shape [1, seq_len, hidden_dim]; got {tuple(beit3_hidden.shape)}."
            )
        if beit3_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"BIOQueryParser hidden dim mismatch: expected {self.hidden_dim}, got {int(beit3_hidden.shape[-1])}."
            )
        seq_len = int(beit3_hidden.shape[1])
        if attention_mask is not None:
            if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (1, seq_len):
                raise ValueError(
                    f"BIOQueryParser expects attention_mask with shape {(1, seq_len)}; got {tuple(attention_mask.shape)}."
                )
        if len(tokens_list) > seq_len:
            raise ValueError(
                f"BIOQueryParser received {len(tokens_list)} tokens for sequence length {seq_len}."
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  Graph-node construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_graph_nodes(
        self,
        query: NormalizedQuery,
        beit3_hidden: Tensor,
        tokens: list[str],
        device: torch.device,
    ) -> tuple[Tensor, list[str]]:
        """Construct four semantic-primitive node embeddings.

        - Node 0 (category):  target first-token embedding + type_embed(0)
        - Node 1 (attribute): mean-pool of attribute-token embeddings + type_embed(1)
        - Node 2 (relation):  relation-type vocabulary embedding + type_embed(2)
        - Node 3 (action):    action-verb vocabulary embedding + type_embed(3)
        """

        def _zero_vec() -> Tensor:
            return torch.zeros(self.hidden_dim, device=device)

        def _type_embed(tid: int) -> Tensor:
            return self.node_type_embed(
                torch.tensor(tid, device=device, dtype=torch.long)
            )

        # -- Node 0: category -------------------------------------------------
        if query["target"]:
            pos = _find_span_positions(query["target"], tokens)
            cat_feat = beit3_hidden[0, pos[0], :] if pos else _zero_vec()
        else:
            cat_feat = _zero_vec()
        node0 = cat_feat + _type_embed(_NODE_CAT)

        # -- Node 1: attribute ------------------------------------------------
        if query["attributes"]:
            all_pos: list[int] = []
            for attr in query["attributes"]:
                all_pos.extend(_find_span_positions(attr, tokens))
            if all_pos:
                attr_feat = beit3_hidden[0, all_pos, :].mean(dim=0)
            else:
                attr_feat = _zero_vec()
        else:
            attr_feat = _zero_vec()
        node1 = attr_feat + _type_embed(_NODE_ATTR)

        # -- Node 2: relation -------------------------------------------------
        if query["relations"]:
            rel = query["relations"][0]
            rid = int(hashlib.md5(rel["type"].encode()).hexdigest(), 16) % 50
            rel_feat = self.relation_embed(
                torch.tensor(rid, device=device, dtype=torch.long)
            )
        else:
            rel_feat = _zero_vec()
        node2 = rel_feat + _type_embed(_NODE_REL)

        # -- Node 3: action ---------------------------------------------------
        if query["actions"]:
            act = query["actions"][0]
            aid = int(hashlib.md5(act["verb"].encode()).hexdigest(), 16) % 50
            act_feat = self.action_embed(
                torch.tensor(aid, device=device, dtype=torch.long)
            )
        else:
            act_feat = _zero_vec()
        node3 = act_feat + _type_embed(_NODE_ACT)

        nodes = torch.stack([node0, node1, node2, node3], dim=0)
        return nodes, list(_NODE_TYPE_NAMES)

    # ══════════════════════════════════════════════════════════════════════════
    #  Rule-based fallback
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _rule_fallback(query_text: str) -> NormalizedQuery:
        """Deterministically normalize fallback text via ``reasonseg.query``."""
        words = [word for word in query_text.strip().lower().split() if word not in _SPECIAL_TOKENS]
        return parse_query(" ".join(words))

    @classmethod
    def _structure_fallback(cls, tokens_list: list[str]) -> NormalizedQuery:
        return cls._rule_fallback(" ".join(tokens_list))


# Backward-compatible alias for older imports.
LLMQueryParser = BIOQueryParser


# ── module-level helpers ─────────────────────────────────────────────────────


def _find_span_positions(text: str, tokens: list[str]) -> list[int]:
    """Return positions of *text* words inside *tokens* (lowercased matching)."""
    words = text.lower().split()
    positions: list[int] = []
    for word in words:
        for i, tok in enumerate(tokens):
            if tok.lower() == word and i not in positions:
                positions.append(i)
                break
    return positions
