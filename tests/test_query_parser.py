from __future__ import annotations

import pytest
import torch

from model.BIOtagging.bio_schema import NormalizedQuery
from model.query_parser import LLMQueryParser, _find_span_positions
from model.vr_ov_types import QueryGraph

_CHECKPOINT = (
    "model/BIOtagging/outputs/"
    "stage1_fast_train50k_plus_mask20k_plus_flashpro2k_20260430_031500/"
    "parser_head_best.pt"
)


def _make_random_hidden(seq_len: int = 8, hidden_dim: int = 768) -> torch.Tensor:
    return torch.randn(1, seq_len, hidden_dim)


def _make_attention_mask(seq_len: int = 8) -> torch.Tensor:
    return torch.ones(1, seq_len)


# ── test_init_with_checkpoint ────────────────────────────────────────────────


class TestInit:
    def test_init_with_checkpoint(self) -> None:
        parser = LLMQueryParser(parser_checkpoint=_CHECKPOINT)
        assert parser.parser_head is not None
        assert parser._has_parser is True
        assert not parser.parser_head.training

    def test_init_without_checkpoint(self) -> None:
        parser = LLMQueryParser(parser_checkpoint=None)
        assert parser._has_parser is False

    def test_init_nonexistent_checkpoint(self) -> None:
        parser = LLMQueryParser(parser_checkpoint="/nonexistent/path.pt")
        assert parser._has_parser is False


# ── test_build_graph_nodes_normal ────────────────────────────────────────────


class TestBuildGraphNodes:
    H = 768

    @staticmethod
    def _make_query(**overrides: object) -> NormalizedQuery:
        base: NormalizedQuery = {
            "target": "cup",
            "attributes": ["red"],
            "relations": [{"type": "on", "target": "table"}],
            "actions": [{"verb": "holding", "target": "spoon"}],
            "negatives": [],
            "exists": True,
        }
        base.update(overrides)  # type: ignore[typeddict-item]
        return base

    @staticmethod
    def _make_parser(**kwargs: object) -> LLMQueryParser:
        return LLMQueryParser(**kwargs)  # type: ignore[arg-type]

    def test_normal_query_4_nodes(self) -> None:
        parser = self._make_parser()
        hidden = _make_random_hidden(8, parser.hidden_dim)
        tokens = ["the", "red", "cup", "on", "table", "holding", "spoon", "sep"]
        query = self._make_query()
        nodes, node_types = parser._build_graph_nodes(
            query, hidden, tokens, hidden.device,
        )
        assert nodes.shape == (4, self.H)
        assert node_types == ["category", "attribute", "relation", "action"]
        assert not torch.isnan(nodes).any()

    def test_empty_query_all_zeros(self) -> None:
        parser = self._make_parser()
        hidden = _make_random_hidden(4, parser.hidden_dim)
        tokens = ["no", "dog", "here", "sep"]
        query = self._make_query(
            target=None, attributes=[], relations=[], actions=[], exists=False,
        )
        nodes, _ = parser._build_graph_nodes(query, hidden, tokens, hidden.device)
        assert nodes.shape == (4, self.H)
        cat_feat = nodes[0]
        assert not torch.allclose(cat_feat, torch.zeros(self.H), atol=1e-6)
        assert not torch.allclose(cat_feat, torch.zeros(self.H), atol=1e-3)

    def test_partial_only_attributes(self) -> None:
        parser = self._make_parser()
        hidden = _make_random_hidden(6, parser.hidden_dim)
        tokens = ["small", "blue", "car", "sep", "pad", "cls"]
        query = self._make_query(
            target="car",
            attributes=["small", "blue"],
            relations=[],
            actions=[],
        )
        nodes, _ = parser._build_graph_nodes(query, hidden, tokens, hidden.device)
        assert nodes.shape == (4, self.H)
        assert not torch.isnan(nodes).any()

    def test_getattr_forward_interface(self) -> None:
        parser = LLMQueryParser(parser_checkpoint=None)
        assert hasattr(parser, "forward")
        assert hasattr(parser, "_has_parser")
        assert hasattr(parser, "parser_head")
        assert hasattr(parser, "gnn")
        assert hasattr(parser, "relation_embed")
        assert hasattr(parser, "action_embed")


# ── test_forward_with_random_hidden ──────────────────────────────────────────


class TestForward:
    def test_forward_with_random_hidden_no_checkpoint(self) -> None:
        parser = LLMQueryParser(parser_checkpoint=None)
        hidden = _make_random_hidden(6, 768)
        mask = _make_attention_mask(6)
        tokens = ["red", "cup", "on", "table", "sep", "cls"]

        result = parser.forward(hidden, mask, tokens)
        assert isinstance(result, QueryGraph)
        assert len(result.nodes) == 4
        assert result.edges.shape == (2, 6)
        assert result.node_types == ["category", "attribute", "relation", "action"]
        for n in result.nodes:
            assert n.shape == (128,)

    def test_forward_short_query(self) -> None:
        parser = LLMQueryParser(parser_checkpoint=None)
        hidden = _make_random_hidden(3, 768)
        mask = _make_attention_mask(3)
        tokens = ["dog", "sep", "cls"]

        result = parser.forward(hidden, mask, tokens)
        assert len(result.nodes) == 4
        for n in result.nodes:
            assert n.shape[0] == 128

    def test_forward_empty_query(self) -> None:
        parser = LLMQueryParser(parser_checkpoint=None)
        hidden = _make_random_hidden(2, 768)
        mask = _make_attention_mask(2)
        tokens: list[str] = []

        result = parser.forward(hidden, mask, tokens)
        assert len(result.nodes) == 4
        for n in result.nodes:
            assert n.shape[0] == 128
        assert not torch.isnan(result.nodes[0]).any()

    def test_forward_with_1024_hidden_no_checkpoint(self) -> None:
        parser = LLMQueryParser(parser_checkpoint=None, hidden_dim=1024)
        hidden = _make_random_hidden(6, 1024)
        mask = _make_attention_mask(6)
        tokens = ["red", "cup", "on", "table", "sep", "cls"]

        result = parser.forward(hidden, mask, tokens)

        assert isinstance(result, QueryGraph)
        assert len(result.nodes) == 4
        for n in result.nodes:
            assert n.shape == (128,)

    def test_forward_fails_loudly_on_hidden_dim_mismatch(self) -> None:
        parser = LLMQueryParser(parser_checkpoint=None)
        hidden = _make_random_hidden(6, 1024)
        mask = _make_attention_mask(6)

        with pytest.raises(ValueError, match="hidden dim mismatch"):
            parser.forward(hidden, mask, ["red", "cup"])

    def test_forward_fails_loudly_when_tokens_exceed_sequence_length(self) -> None:
        parser = LLMQueryParser(parser_checkpoint=None)
        hidden = _make_random_hidden(2, 768)
        mask = _make_attention_mask(2)

        with pytest.raises(ValueError, match="received 3 tokens"):
            parser.forward(hidden, mask, ["red", "cup", "extra"])


# ── test_rule_fallback ───────────────────────────────────────────────────────


class TestRuleFallback:
    def test_red_cup(self) -> None:
        result = LLMQueryParser._rule_fallback("red cup")
        assert result["target"] == "cup"
        assert result["attributes"] == ["red"]
        assert result["exists"] is True

    def test_single_word(self) -> None:
        result = LLMQueryParser._rule_fallback("dog")
        assert result["target"] == "dog"
        assert result["attributes"] == []
        assert result["exists"] is True

    def test_empty_string(self) -> None:
        result = LLMQueryParser._rule_fallback("")
        assert result["target"] is None
        assert result["exists"] is False
        assert "empty_query" in result["negatives"]

    def test_multiple_attributes(self) -> None:
        result = LLMQueryParser._rule_fallback("small blue car")
        assert result["target"] == "car"
        assert result["attributes"] == ["small", "blue"]


# ── test_find_span_positions ─────────────────────────────────────────────────


class TestFindSpanPositions:
    def test_single_word(self) -> None:
        tokens = ["the", "red", "cup", "sep"]
        assert _find_span_positions("cup", tokens) == [2]
        assert _find_span_positions("red", tokens) == [1]

    def test_no_match(self) -> None:
        tokens = ["the", "red", "cup"]
        assert _find_span_positions("green", tokens) == []

    def test_case_insensitive(self) -> None:
        tokens = ["The", "Red", "Cup"]
        assert _find_span_positions("cup", tokens) == [2]
