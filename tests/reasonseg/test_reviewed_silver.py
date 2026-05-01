from __future__ import annotations

from model.BIOtagging.reviewed_silver import finalize_reviewed_pairs, merge_silver_pairs, normalize_query_text


def _q(target: str | None, exists: bool = True) -> dict:
    return {
        "target": target,
        "attributes": [],
        "relations": [],
        "actions": [],
        "negatives": [] if exists else ["absent_object"],
        "exists": exists,
    }


def test_normalize_query_text_collapses_case_and_space() -> None:
    assert normalize_query_text("  Red   Cup ") == "red cup"


def test_merge_silver_pairs_overlays_duplicates() -> None:
    base = [("red cup", _q("cup")), ("dog", _q("dog"))]
    overlay = [("Red   Cup", _q("mug")), ("cat", _q("cat"))]
    merged = merge_silver_pairs(base, overlay)
    merged_map = {normalize_query_text(q): s for q, s in merged}
    assert merged_map["red cup"]["target"] == "mug"
    assert merged_map["dog"]["target"] == "dog"
    assert merged_map["cat"]["target"] == "cat"


def test_finalize_reviewed_pairs_uses_candidate_when_passed() -> None:
    seed = [("red cup", _q("cup"))]
    candidate = [_q("mug")]
    reviewed = [True]
    finalized = finalize_reviewed_pairs(seed, candidate, reviewed)
    assert finalized[0][1]["target"] == "mug"


def test_finalize_reviewed_pairs_falls_back_when_failed() -> None:
    seed = [("red cup", _q("cup"))]
    candidate = [_q("mug")]
    reviewed = [False]
    finalized = finalize_reviewed_pairs(seed, candidate, reviewed)
    assert finalized[0][1]["target"] == "cup"


def test_finalize_reviewed_pairs_skips_when_no_fallback() -> None:
    seed = [("red cup", _q("cup"))]
    candidate = [None]
    reviewed = [False]
    finalized = finalize_reviewed_pairs(seed, candidate, reviewed, fallback_to_seed=False)
    assert finalized == []
