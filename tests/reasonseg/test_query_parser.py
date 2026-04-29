import pytest
from typing import TypedDict

from reasonseg.query import parse_query


class RelationEntry(TypedDict):
    type: str
    target: str


class ActionEntry(TypedDict):
    verb: str
    target: str | None


class NormalizedQuery(TypedDict):
    target: str | None
    attributes: list[str]
    relations: list[RelationEntry]
    actions: list[ActionEntry]
    negatives: list[str]
    exists: bool


class QueryCase(TypedDict):
    name: str
    query: str
    expected: NormalizedQuery


EXPECTED_SCHEMA_KEYS = (
    "target",
    "attributes",
    "relations",
    "actions",
    "negatives",
    "exists",
)


def assert_normalized_query_contract(normalized_query: NormalizedQuery) -> None:
    assert tuple(normalized_query.keys()) == EXPECTED_SCHEMA_KEYS
    assert normalized_query["target"] is None or isinstance(
        normalized_query["target"], str
    )
    assert isinstance(normalized_query["attributes"], list)
    assert all(
        isinstance(attribute, str) for attribute in normalized_query["attributes"]
    )
    assert isinstance(normalized_query["relations"], list)
    assert isinstance(normalized_query["actions"], list)
    assert isinstance(normalized_query["negatives"], list)
    assert all(isinstance(flag, str) for flag in normalized_query["negatives"])
    assert isinstance(normalized_query["exists"], bool)

    for relation in normalized_query["relations"]:
        assert tuple(relation.keys()) == ("type", "target")
        assert isinstance(relation["type"], str)
        assert isinstance(relation["target"], str)

    for action in normalized_query["actions"]:
        assert tuple(action.keys()) == ("verb", "target")
        assert isinstance(action["verb"], str)
        assert action["target"] is None or isinstance(action["target"], str)


def _collect_cases(
    normalized_query_cases: list[QueryCase], prefix: str
) -> list[QueryCase]:
    return [case for case in normalized_query_cases if case["name"].startswith(prefix)]


def test_all_cases_expose_the_expected_schema(
    normalized_query_cases: list[QueryCase],
) -> None:
    assert len(normalized_query_cases) >= 12

    for case in normalized_query_cases:
        assert_normalized_query_contract(case["expected"])


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param(
            "dog",
            {
                "target": "dog",
                "attributes": [],
                "relations": [],
                "actions": [],
                "negatives": [],
                "exists": True,
            },
            id="dog",
        ),
        pytest.param(
            "person",
            {
                "target": "person",
                "attributes": [],
                "relations": [],
                "actions": [],
                "negatives": [],
                "exists": True,
            },
            id="person",
        ),
        pytest.param(
            "red dress",
            {
                "target": "dress",
                "attributes": ["red"],
                "relations": [],
                "actions": [],
                "negatives": [],
                "exists": True,
            },
            id="red-dress",
        ),
        pytest.param(
            "wooden table",
            {
                "target": "table",
                "attributes": ["wooden"],
                "relations": [],
                "actions": [],
                "negatives": [],
                "exists": True,
            },
            id="wooden-table",
        ),
        pytest.param(
            "small blue car",
            {
                "target": "car",
                "attributes": ["small", "blue"],
                "relations": [],
                "actions": [],
                "negatives": [],
                "exists": True,
            },
            id="small-blue-car",
        ),
        pytest.param(
            "dog behind bicycle",
            {
                "target": "dog",
                "attributes": [],
                "relations": [{"type": "behind", "target": "bicycle"}],
                "actions": [],
                "negatives": [],
                "exists": True,
            },
            id="dog-behind-bicycle",
        ),
        pytest.param(
            "cup on table",
            {
                "target": "cup",
                "attributes": [],
                "relations": [{"type": "on", "target": "table"}],
                "actions": [],
                "negatives": [],
                "exists": True,
            },
            id="cup-on-table",
        ),
        pytest.param(
            "woman with hat",
            {
                "target": "woman",
                "attributes": [],
                "relations": [{"type": "with", "target": "hat"}],
                "actions": [],
                "negatives": [],
                "exists": True,
            },
            id="woman-with-hat",
        ),
        pytest.param(
            "dog running",
            {
                "target": "dog",
                "attributes": [],
                "relations": [],
                "actions": [{"verb": "running", "target": None}],
                "negatives": [],
                "exists": True,
            },
            id="dog-running",
        ),
        pytest.param(
            "man watering flowers",
            {
                "target": "man",
                "attributes": [],
                "relations": [],
                "actions": [{"verb": "watering", "target": "flowers"}],
                "negatives": [],
                "exists": True,
            },
            id="man-watering-flowers",
        ),
        pytest.param(
            "it",
            {
                "target": None,
                "attributes": [],
                "relations": [],
                "actions": [],
                "negatives": ["missing_target"],
                "exists": False,
            },
            id="pronoun-only",
        ),
        pytest.param(
            "behind the table",
            {
                "target": None,
                "attributes": [],
                "relations": [{"type": "behind", "target": "table"}],
                "actions": [],
                "negatives": ["missing_target"],
                "exists": False,
            },
            id="relation-only",
        ),
        pytest.param(
            "",
            {
                "target": None,
                "attributes": [],
                "relations": [],
                "actions": [],
                "negatives": ["empty_query"],
                "exists": False,
            },
            id="empty-query",
        ),
        pytest.param(
            "???",
            {
                "target": None,
                "attributes": [],
                "relations": [],
                "actions": [],
                "negatives": ["malformed_query"],
                "exists": False,
            },
            id="malformed-query",
        ),
    ],
)
def test_golden_normalized_queries(
    query: str,
    expected: NormalizedQuery,
) -> None:
    actual = parse_query(query)
    assert actual == expected


def test_case_matrix_covers_required_reasoning_modes(
    normalized_query_cases: list[QueryCase],
) -> None:
    noun_cases = _collect_cases(normalized_query_cases, "noun_")
    attribute_cases = _collect_cases(normalized_query_cases, "attribute_")
    relation_cases = _collect_cases(normalized_query_cases, "relation_")
    action_cases = _collect_cases(normalized_query_cases, "action_")
    no_target_cases = _collect_cases(normalized_query_cases, "no_target_")
    negative_cases = _collect_cases(normalized_query_cases, "negative_")

    assert len(noun_cases) >= 2
    assert len(attribute_cases) >= 3
    assert len(relation_cases) >= 3
    assert len(action_cases) >= 2
    assert len(no_target_cases) >= 2
    assert len(negative_cases) >= 2


def test_nonexistent_targets_are_marked_with_exists_false(
    normalized_query_cases: list[QueryCase],
) -> None:
    no_target_cases = _collect_cases(normalized_query_cases, "no_target_")

    for case in no_target_cases:
        expected = case["expected"]
        assert expected["target"] is None
        assert expected["exists"] is False
        assert "missing_target" in expected["negatives"]


def test_empty_and_malformed_queries_fail_in_a_controlled_way(
    normalized_query_cases: list[QueryCase],
) -> None:
    empty_case = next(
        case
        for case in normalized_query_cases
        if case["name"] == "negative_empty_query"
    )
    malformed_case = next(
        case
        for case in normalized_query_cases
        if case["name"] == "negative_malformed_query"
    )

    assert empty_case["expected"] == {
        "target": None,
        "attributes": [],
        "relations": [],
        "actions": [],
        "negatives": ["empty_query"],
        "exists": False,
    }
    assert malformed_case["expected"] == {
        "target": None,
        "attributes": [],
        "relations": [],
        "actions": [],
        "negatives": ["malformed_query"],
        "exists": False,
    }
