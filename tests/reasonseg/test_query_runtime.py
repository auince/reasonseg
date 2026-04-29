# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false
from __future__ import annotations

from typing import TypedDict

import pytest

from reasonseg.query import compose_query_score, parse_query


class ScoreCase(TypedDict):
    name: str
    query: str
    signals: dict[str, object]
    expected: dict[str, object]


def test_parser_matches_task2_golden_contract(
    normalized_query_cases: list[dict[str, object]],
) -> None:
    for case in normalized_query_cases:
        query = case["query"]
        expected = case["expected"]
        assert parse_query(query) == expected


@pytest.mark.parametrize(
    ("query", "expected_negative"),
    [
        pytest.param("no dog", "absent_object", id="no-dog"),
        pytest.param("without bicycle", "absent_object", id="without-bicycle"),
    ],
)
def test_absent_object_style_queries_reject_explicitly(
    query: str,
    expected_negative: str,
) -> None:
    normalized = parse_query(query)

    score = compose_query_score(
        normalized,
        {"target": 0.9, "attributes": {"red": 0.9}, "relations": {}, "actions": {}},
    )

    assert normalized["exists"] is False
    assert expected_negative in normalized["negatives"]
    assert score.rejected is True
    assert score.rejection_reason == expected_negative
    assert score.overall_score == 0.0


@pytest.mark.parametrize(
    "score_case",
    [
        {
            "name": "attribute_relation_action_weighted_average",
            "query": "small blue car",
            "signals": {
                "target": 0.8,
                "attributes": {"small": 0.6, "blue": 1.0},
                "relations": {},
                "actions": {},
            },
            "expected": {
                "target_score": 0.8,
                "attribute_score": 0.8,
                "relation_score": 1.0,
                "action_score": 1.0,
                "overall_score": 0.86,
                "rejected": False,
                "rejection_reason": None,
            },
        },
        {
            "name": "full_composition_uses_missing_signal_as_zero",
            "query": "man watering flowers",
            "signals": {
                "target": 0.9,
                "attributes": {},
                "relations": {},
                "actions": {"watering::flowers": 0.4},
            },
            "expected": {
                "target_score": 0.9,
                "attribute_score": 1.0,
                "relation_score": 1.0,
                "action_score": 0.4,
                "overall_score": 0.855,
                "rejected": False,
                "rejection_reason": None,
            },
        },
        {
            "name": "relation_contributes_to_composed_score",
            "query": "dog behind bicycle",
            "signals": {
                "target": 0.7,
                "attributes": {},
                "relations": {"behind::bicycle": 0.2},
                "actions": {},
            },
            "expected": {
                "target_score": 0.7,
                "attribute_score": 1.0,
                "relation_score": 0.2,
                "action_score": 1.0,
                "overall_score": 0.715,
                "rejected": False,
                "rejection_reason": None,
            },
        },
    ],
    ids=[
        "attribute_relation_action_weighted_average",
        "full_composition_uses_missing_signal_as_zero",
        "relation_contributes_to_composed_score",
    ],
)
def test_compositional_scoring_cases(score_case: ScoreCase) -> None:
    normalized = parse_query(score_case["query"])
    score = compose_query_score(normalized, score_case["signals"])

    assert score.target_score == pytest.approx(score_case["expected"]["target_score"])
    assert score.attribute_score == pytest.approx(
        score_case["expected"]["attribute_score"]
    )
    assert score.relation_score == pytest.approx(
        score_case["expected"]["relation_score"]
    )
    assert score.action_score == pytest.approx(score_case["expected"]["action_score"])
    assert score.overall_score == pytest.approx(score_case["expected"]["overall_score"])
    assert score.rejected is score_case["expected"]["rejected"]
    assert score.rejection_reason == score_case["expected"]["rejection_reason"]
