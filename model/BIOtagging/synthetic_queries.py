from __future__ import annotations

import json
from pathlib import Path

from .bio_schema import NormalizedQuery

_TARGETS = [
    "cup", "dog", "person", "car", "table", "chair", "phone", "laptop",
    "bicycle", "bottle", "book", "cat", "bird", "horse", "truck", "plate",
    "spoon", "fork", "bowl", "apple", "banana", "pizza", "cake", "bus",
    "train", "bear", "zebra", "giraffe", "elephant", "mouse", "keyboard",
    "monitor", "backpack", "umbrella", "surfboard", "tennis racket",
    "baseball bat", "skis", "snowboard", "suitcase", "vase", "lamp",
    "clock", "mirror", "window", "door", "pillow", "blanket",
]

_ATTRIBUTES = [
    "red", "blue", "green", "yellow", "black", "white", "orange", "purple",
    "large", "small", "tall", "short", "wide", "narrow", "big", "tiny",
    "wooden", "metal", "plastic", "glass", "ceramic", "leather", "fabric",
    "round", "square", "rectangular", "flat", "shiny", "matte", "striped",
    "checked", "dotted", "patterned", "dark", "light", "bright", "pale",
]

_RELATIONS = [
    "behind", "on", "under", "beside", "next to", "in front of", "above",
    "below", "inside", "outside", "with", "near", "across from", "left of",
    "right of", "between", "atop", "against",
]

_ACTION_VERBS = [
    "holding", "carrying", "eating", "drinking", "riding", "driving",
    "wearing", "sitting on", "standing on", "pointing at", "looking at",
    "touching", "pushing", "pulling", "covering", "watering",
]

_ACTION_TARGETS = [
    "apple", "book", "phone", "bottle", "cup", "bag", "hat",
    "ball", "umbrella", "laptop", "flower", "box", "toy",
]

_ABSENT_PREFIXES = ["no", "without"]


def generate_synthetic_queries(count: int = 5000) -> list[NormalizedQuery]:
    import random
    random.seed(42)

    queries: list[NormalizedQuery] = []
    for _ in range(count):
        query_type = random.random()

        if query_type < 0.05:
            queries.append(_absent_query(random))
        elif query_type < 0.10:
            queries.append(_pronoun_query(random))
        elif query_type < 0.30:
            queries.append(_noun_only_query(random))
        elif query_type < 0.50:
            queries.append(_attribute_query(random))
        elif query_type < 0.75:
            queries.append(_relation_query(random))
        else:
            queries.append(_action_query(random))

    return queries


def _absent_query(rng: random.Random) -> NormalizedQuery:
    target = rng.choice(_TARGETS)
    prefix = rng.choice(_ABSENT_PREFIXES)
    return {
        "target": None,
        "attributes": [],
        "relations": [],
        "actions": [],
        "negatives": ["absent_object"],
        "exists": False,
    }


def _pronoun_query(_rng: random.Random) -> NormalizedQuery:
    return {
        "target": None,
        "attributes": [],
        "relations": [],
        "actions": [],
        "negatives": ["missing_target"],
        "exists": False,
    }


def _noun_only_query(rng: random.Random) -> NormalizedQuery:
    target = rng.choice(_TARGETS)
    return {
        "target": target,
        "attributes": [],
        "relations": [],
        "actions": [],
        "negatives": [],
        "exists": True,
    }


def _attribute_query(rng: random.Random) -> NormalizedQuery:
    target = rng.choice(_TARGETS)
    num_attrs = rng.randint(1, 3)
    attrs = list(set(rng.choices(_ATTRIBUTES, k=num_attrs)))
    return {
        "target": target,
        "attributes": attrs,
        "relations": [],
        "actions": [],
        "negatives": [],
        "exists": True,
    }


def _relation_query(rng: random.Random) -> NormalizedQuery:
    target = rng.choice(_TARGETS)
    num_attrs = rng.randint(0, 2)
    attrs = list(set(rng.choices(_ATTRIBUTES, k=num_attrs))) if num_attrs > 0 else []

    num_rels = rng.randint(1, 2)
    relations = []
    for _ in range(num_rels):
        rel_type = rng.choice(_RELATIONS)
        rel_target = rng.choice(_TARGETS)
        relations.append({"type": rel_type, "target": rel_target})

    return {
        "target": target,
        "attributes": attrs,
        "relations": relations,
        "actions": [],
        "negatives": [],
        "exists": True,
    }


def _action_query(rng: random.Random) -> NormalizedQuery:
    target = rng.choice(_TARGETS)
    num_attrs = rng.randint(0, 2)
    attrs = list(set(rng.choices(_ATTRIBUTES, k=num_attrs))) if num_attrs > 0 else []

    verb = rng.choice(_ACTION_VERBS)
    has_target = rng.random() < 0.7
    act_target: str | None = rng.choice(_ACTION_TARGETS) if has_target else None

    return {
        "target": target,
        "attributes": attrs,
        "relations": [],
        "actions": [{"verb": verb, "target": act_target}],
        "negatives": [],
        "exists": True,
    }


def export_queries_to_file(
    queries: list[NormalizedQuery],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
