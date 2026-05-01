from __future__ import annotations

from .bio_schema import NormalizedQuery, tokens_to_bio_labels
from .review import _classify_query
from reasonseg.modeling.prompting import infer_slice_tag

TYPE_BASE_SCORE: dict[str, float] = {
    "noun": 0.0,
    "attribute": 1.0,
    "relation": 2.0,
    "action": 3.0,
    "absent": 0.0,
}

_SCALE = 100.0


def count_non_o_labels(raw_query: str, structure: NormalizedQuery) -> int:
    tokens = raw_query.lower().split()
    labels = tokens_to_bio_labels(tokens, structure)
    return sum(1 for label in labels if label != 0)


def score_complexity(raw_query: str, structure: NormalizedQuery) -> float:
    tier = _classify_query(structure)
    base = TYPE_BASE_SCORE.get(tier, 0.0)
    return base * _SCALE + float(count_non_o_labels(raw_query, structure))


def classify_tier(structure: NormalizedQuery) -> str:
    return _classify_query(structure)


def classify_slice_tag(structure: NormalizedQuery) -> str:
    return infer_slice_tag(structure)


def select_top_complex(
    items: list[tuple[str, NormalizedQuery]],
    top_k: int,
) -> list[tuple[str, NormalizedQuery]]:
    scored = [(score_complexity(q, s), q, s) for q, s in items]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(q, s) for _, q, s in scored[:top_k]]
