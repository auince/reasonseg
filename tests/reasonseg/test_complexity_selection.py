from __future__ import annotations

import pytest

from model.BIOtagging.bio_schema import NormalizedQuery
from model.BIOtagging.complexity_selection import (
    TYPE_BASE_SCORE,
    classify_slice_tag,
    classify_tier,
    count_non_o_labels,
    score_complexity,
    select_top_complex,
)


def _q(**overrides: object) -> NormalizedQuery:
    defaults: NormalizedQuery = {
        "target": None,
        "attributes": [],
        "relations": [],
        "actions": [],
        "negatives": [],
        "exists": True,
    }
    merged = {**defaults, **overrides}  # type: ignore[misc]
    return merged  # type: ignore[return-value]


NOUN_DOG: tuple[str, NormalizedQuery] = ("dog", _q(target="dog"))
ATTR_RED_DRESS: tuple[str, NormalizedQuery] = (
    "red dress",
    _q(target="dress", attributes=["red"]),
)
REL_DOG_BEHIND: tuple[str, NormalizedQuery] = (
    "dog behind bicycle",
    _q(target="dog", relations=[{"type": "behind", "target": "bicycle"}]),
)
ACT_MAN_WATERING: tuple[str, NormalizedQuery] = (
    "man watering flowers",
    _q(
        target="man",
        actions=[{"verb": "watering", "target": "flowers"}],
    ),
)
ABSENT_IT: tuple[str, NormalizedQuery] = (
    "it",
    _q(target=None, negatives=["missing_target"], exists=False),
)

ALL_PAIRS = [NOUN_DOG, ATTR_RED_DRESS, REL_DOG_BEHIND, ACT_MAN_WATERING, ABSENT_IT]


class TestComplexityScoreOrdering:
    def test_action_gt_relation(self) -> None:
        assert score_complexity(*ACT_MAN_WATERING) > score_complexity(*REL_DOG_BEHIND)

    def test_relation_gt_attribute(self) -> None:
        assert score_complexity(*REL_DOG_BEHIND) > score_complexity(*ATTR_RED_DRESS)

    def test_attribute_gt_noun(self) -> None:
        assert score_complexity(*ATTR_RED_DRESS) > score_complexity(*NOUN_DOG)

    def test_absent_score_low(self) -> None:
        assert score_complexity(*ABSENT_IT) < score_complexity(*NOUN_DOG)

    def test_absent_scores_zeroish(self) -> None:
        assert score_complexity(*ABSENT_IT) < 10.0

    @pytest.mark.parametrize(
        "pair,tier",
        [
            (NOUN_DOG, "noun"),
            (ATTR_RED_DRESS, "attribute"),
            (REL_DOG_BEHIND, "relation"),
            (ACT_MAN_WATERING, "action"),
            (ABSENT_IT, "absent"),
        ],
    )
    def test_score_falls_in_expected_tier_range(
        self, pair: tuple[str, NormalizedQuery], tier: str
    ) -> None:
        base = TYPE_BASE_SCORE[tier]
        score = score_complexity(*pair)
        assert base * 100.0 <= score < (base + 1.0) * 100.0


class TestSelectTopComplex:
    def test_selects_top_k(self) -> None:
        result = select_top_complex(ALL_PAIRS, 3)
        assert len(result) == 3

    def test_top_three_are_most_complex(self) -> None:
        result = select_top_complex(ALL_PAIRS, 3)
        tiers = [classify_tier(s) for _, s in result]
        assert tiers == ["action", "relation", "attribute"]

    def test_select_top_k_larger_than_input_returns_all_sorted(self) -> None:
        result = select_top_complex(ALL_PAIRS, 100)
        assert len(result) == len(ALL_PAIRS)
        scores = [score_complexity(q, s) for q, s in result]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_empty_input(self) -> None:
        result = select_top_complex([], 5)
        assert result == []


class TestClassifyTier:
    @pytest.mark.parametrize(
        "pair,expected_tier",
        [
            (NOUN_DOG, "noun"),
            (ATTR_RED_DRESS, "attribute"),
            (REL_DOG_BEHIND, "relation"),
            (ACT_MAN_WATERING, "action"),
            (ABSENT_IT, "absent"),
        ],
    )
    def test_tier_classification(
        self, pair: tuple[str, NormalizedQuery], expected_tier: str
    ) -> None:
        assert classify_tier(pair[1]) == expected_tier


class TestClassifySliceTag:
    @pytest.mark.parametrize(
        "pair,expected_slice",
        [
            (NOUN_DOG, "noun"),
            (ATTR_RED_DRESS, "attribute"),
            (REL_DOG_BEHIND, "relation_action"),
            (ACT_MAN_WATERING, "relation_action"),
            (ABSENT_IT, "no_target"),
        ],
    )
    def test_slice_tag_classification(
        self, pair: tuple[str, NormalizedQuery], expected_slice: str
    ) -> None:
        assert classify_slice_tag(pair[1]) == expected_slice


class TestCountNonOLabels:
    def test_noun_one_tag(self) -> None:
        assert count_non_o_labels(*NOUN_DOG) == 1

    def test_attribute_two_tags(self) -> None:
        assert count_non_o_labels(*ATTR_RED_DRESS) == 2

    def test_relation_three_tags(self) -> None:
        assert count_non_o_labels(*REL_DOG_BEHIND) == 3

    def test_action_three_tags(self) -> None:
        assert count_non_o_labels(*ACT_MAN_WATERING) == 3

    def test_absent_zero_tags(self) -> None:
        assert count_non_o_labels(*ABSENT_IT) == 0

    def test_multiple_attributes(self) -> None:
        struct = _q(target="car", attributes=["small", "blue", "fast"])
        count = count_non_o_labels("small blue fast car", struct)
        assert count == 4

    def test_multiple_relations(self) -> None:
        struct = _q(
            target="dog",
            relations=[
                {"type": "behind", "target": "bicycle"},
                {"type": "near", "target": "tree"},
            ],
        )
        count = count_non_o_labels("dog behind bicycle near tree", struct)
        assert count == 5
