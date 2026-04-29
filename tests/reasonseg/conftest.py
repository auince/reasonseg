import copy
from typing import TypedDict

import pytest


class RelationEntry(TypedDict):
    type: str
    target: str


class ActionEntry(TypedDict):
    verb: str
    target: str | None


class QueryCase(TypedDict):
    name: str
    query: str
    expected: dict[str, object]


def _case(
    name: str,
    query: str,
    *,
    target: str | None,
    attributes: list[str] | None = None,
    relations: list[RelationEntry] | None = None,
    actions: list[ActionEntry] | None = None,
    negatives: list[str] | None = None,
    exists: bool,
) -> QueryCase:
    return {
        "name": name,
        "query": query,
        "expected": {
            "target": target,
            "attributes": attributes or [],
            "relations": relations or [],
            "actions": actions or [],
            "negatives": negatives or [],
            "exists": exists,
        },
    }


@pytest.fixture(scope="session")
def normalized_query_cases() -> list[QueryCase]:
    cases: list[QueryCase] = [
        _case("noun_dog", "dog", target="dog", exists=True),
        _case("noun_person", "person", target="person", exists=True),
        _case(
            "attribute_red_dress",
            "red dress",
            target="dress",
            attributes=["red"],
            exists=True,
        ),
        _case(
            "attribute_wooden_table",
            "wooden table",
            target="table",
            attributes=["wooden"],
            exists=True,
        ),
        _case(
            "attribute_small_blue_car",
            "small blue car",
            target="car",
            attributes=["small", "blue"],
            exists=True,
        ),
        _case(
            "relation_dog_behind_bicycle",
            "dog behind bicycle",
            target="dog",
            relations=[{"type": "behind", "target": "bicycle"}],
            exists=True,
        ),
        _case(
            "relation_cup_on_table",
            "cup on table",
            target="cup",
            relations=[{"type": "on", "target": "table"}],
            exists=True,
        ),
        _case(
            "relation_woman_with_hat",
            "woman with hat",
            target="woman",
            relations=[{"type": "with", "target": "hat"}],
            exists=True,
        ),
        _case(
            "action_dog_running",
            "dog running",
            target="dog",
            actions=[{"verb": "running", "target": None}],
            exists=True,
        ),
        _case(
            "action_man_watering_flowers",
            "man watering flowers",
            target="man",
            actions=[{"verb": "watering", "target": "flowers"}],
            exists=True,
        ),
        _case(
            "no_target_pronoun_only",
            "it",
            target=None,
            negatives=["missing_target"],
            exists=False,
        ),
        _case(
            "no_target_relation_only",
            "behind the table",
            target=None,
            relations=[{"type": "behind", "target": "table"}],
            negatives=["missing_target"],
            exists=False,
        ),
        _case(
            "negative_empty_query",
            "",
            target=None,
            negatives=["empty_query"],
            exists=False,
        ),
        _case(
            "negative_malformed_query",
            "???",
            target=None,
            negatives=["malformed_query"],
            exists=False,
        ),
    ]
    return copy.deepcopy(cases)
