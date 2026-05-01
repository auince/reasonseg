from __future__ import annotations

from collections import Counter

from .bio_schema import NormalizedQuery
from .complexity_selection import classify_tier


def normalize_query_text(query: str) -> str:
    return " ".join(query.lower().split())


def merge_silver_pairs(
    base_pairs: list[tuple[str, NormalizedQuery]],
    overlay_pairs: list[tuple[str, NormalizedQuery]],
) -> list[tuple[str, NormalizedQuery]]:
    merged: dict[str, tuple[str, NormalizedQuery]] = {}
    for query, structure in base_pairs:
        merged[normalize_query_text(query)] = (query, structure)
    for query, structure in overlay_pairs:
        merged[normalize_query_text(query)] = (query, structure)
    return list(merged.values())


def finalize_reviewed_pairs(
    seed_pairs: list[tuple[str, NormalizedQuery]],
    candidate_structures: list[NormalizedQuery | None],
    review_results: list[bool],
    *,
    fallback_to_seed: bool = True,
) -> list[tuple[str, NormalizedQuery]]:
    if len(seed_pairs) != len(candidate_structures) or len(seed_pairs) != len(review_results):
        raise ValueError("seed_pairs, candidate_structures, and review_results must align")
    finalized: list[tuple[str, NormalizedQuery]] = []
    for (query, seed_structure), candidate, passed in zip(
        seed_pairs, candidate_structures, review_results
    ):
        if candidate is not None and passed:
            finalized.append((query, candidate))
        elif fallback_to_seed:
            finalized.append((query, seed_structure))
    return finalized


def tier_counts(pairs: list[tuple[str, NormalizedQuery]]) -> dict[str, int]:
    return dict(Counter(classify_tier(structure) for _, structure in pairs))
