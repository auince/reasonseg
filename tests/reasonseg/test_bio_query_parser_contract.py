from __future__ import annotations

import torch

from model.query_parser import BIOQueryParser, LLMQueryParser
from reasonseg.query import parse_query


def _make_hidden(seq_len: int, hidden_dim: int) -> torch.Tensor:
    return torch.randn(1, seq_len, hidden_dim)


def _make_attention_mask(seq_len: int) -> torch.Tensor:
    return torch.ones(1, seq_len)


def test_legacy_alias_points_to_canonical_bio_parser() -> None:
    assert LLMQueryParser is BIOQueryParser


def test_missing_checkpoint_uses_deterministic_normalized_fallback() -> None:
    parser = BIOQueryParser(parser_checkpoint=None)
    tokens = ["[CLS]", "the", "red", "dress", "[SEP]"]
    hidden = _make_hidden(len(tokens), parser.hidden_dim)
    attention_mask = _make_attention_mask(len(tokens))

    normalized = parser.decode_to_structure(hidden, attention_mask, tokens)

    assert parser._has_parser is False
    assert parser.parser_mode == "rule_fallback"
    assert normalized == parse_query("the red dress")


def test_nonexistent_checkpoint_uses_same_fallback_contract() -> None:
    parser = BIOQueryParser(parser_checkpoint="/tmp/does-not-exist-parser-head.pt")
    tokens = ["[CLS]", "behind", "the", "table", "[SEP]"]
    hidden = _make_hidden(len(tokens), parser.hidden_dim)
    attention_mask = _make_attention_mask(len(tokens))

    normalized = parser.decode_to_structure(hidden, attention_mask, tokens)

    assert parser._has_parser is False
    assert parser.parser_mode == "rule_fallback"
    assert normalized == {
        "target": None,
        "attributes": [],
        "relations": [{"type": "behind", "target": "table"}],
        "actions": [],
        "negatives": ["missing_target"],
        "exists": False,
    }


def test_fallback_rule_parser_reuses_normalized_query_schema() -> None:
    normalized = BIOQueryParser._rule_fallback("no dog")

    assert normalized == parse_query("no dog")
