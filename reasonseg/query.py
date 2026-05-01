from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from collections.abc import Mapping
from typing import TypedDict, cast


CANONICAL_QUERY_KEYS = (
    "target",
    "attributes",
    "relations",
    "actions",
    "negatives",
    "exists",
)

_RELATION_WORDS = {
    "behind",
    "on",
    "with",
    # spatial (single-word from synthetic_queries.py)
    "under",
    "beside",
    "above",
    "below",
    "inside",
    "outside",
    "near",
    "against",
    "atop",
    "between",
}
_ATTRIBUTE_WORDS = {
    # colors
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    # size / shape
    "small",
    "large",
    "big",
    "tiny",
    "tall",
    "short",
    "wide",
    "narrow",
    # material
    "wooden",
    "metal",
    "plastic",
    "glass",
    "ceramic",
    "leather",
    "fabric",
    # shape / pattern
    "round",
    "square",
    "rectangular",
    "flat",
    "striped",
    "checked",
    "dotted",
    "patterned",
    # descriptors
    "shiny",
    "dark",
    "light",
    "bright",
    "pale",
}
_ACTION_TARGETS = {
    "watering",
    # transitive single-word from synthetic_queries.py
    "holding",
    "carrying",
    "eating",
    "drinking",
    "riding",
    "driving",
    "wearing",
    "touching",
    "pushing",
    "pulling",
    "covering",
}
_ACTION_STANDALONE = {
    "running",
    "standing",
    "sitting",
    "walking",
    "smiling",
}
_ABSENT_PREFIXES = {"no", "without", "absent"}
_PRONOUN_TARGETS = {"it", "they", "them", "this", "that", "these", "those"}
_LEADING_ARTICLES = {"the", "a", "an"}
_MALFORMED_QUERY_RE = re.compile(r"^[^\w]+$")


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


class ParsedTarget(TypedDict):
    target: str
    attributes: list[str]


class ParsedRelation(TypedDict):
    subject: list[str]
    type: str
    target: str


class ParsedAction(TypedDict):
    subject: list[str]
    verb: str
    target: str | None


@dataclass(frozen=True)
class CompositionalScore:
    target_score: float
    attribute_score: float
    relation_score: float
    action_score: float
    overall_score: float
    rejected: bool
    rejection_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class QueryParser:
    def parse(self, query: str) -> NormalizedQuery:
        return _parse_query(query)


def parse_query(query: str) -> NormalizedQuery:
    return QueryParser().parse(query)


def compose_query_score(
    query: NormalizedQuery,
    signals: Mapping[str, object],
    *,
    target_weight: float = 0.55,
    attribute_weight: float = 0.15,
    relation_weight: float = 0.15,
    action_weight: float = 0.15,
) -> CompositionalScore:
    negatives = {item.lower() for item in query["negatives"]}
    if query["exists"] is False:
        reason = "absent_object" if "absent_object" in negatives else "missing_target"
        return CompositionalScore(
            target_score=0.0,
            attribute_score=0.0,
            relation_score=0.0,
            action_score=0.0,
            overall_score=0.0,
            rejected=True,
            rejection_reason=reason,
        )

    target_score = _score_scalar(signals.get("target"))
    attribute_score = _mean_constraint_scores(
        query["attributes"], _score_lookup(signals.get("attributes"))
    )
    relation_score = _mean_constraint_scores(
        [_relation_key(item) for item in query["relations"]],
        _score_lookup(signals.get("relations")),
    )
    action_score = _mean_constraint_scores(
        [_action_key(item) for item in query["actions"]],
        _score_lookup(signals.get("actions")),
    )

    overall_score = (
        (target_score * target_weight)
        + (attribute_score * attribute_weight)
        + (relation_score * relation_weight)
        + (action_score * action_weight)
    )
    return CompositionalScore(
        target_score=target_score,
        attribute_score=attribute_score,
        relation_score=relation_score,
        action_score=action_score,
        overall_score=overall_score,
        rejected=False,
        rejection_reason=None,
    )


def _parse_query(raw_query: str) -> NormalizedQuery:
    query = raw_query.strip().lower()
    if not query:
        return _normalized_query(target=None, negatives=["empty_query"], exists=False)
    if _MALFORMED_QUERY_RE.fullmatch(query):
        return _normalized_query(
            target=None, negatives=["malformed_query"], exists=False
        )

    tokens = query.split()
    absent_query = _extract_absent_query(tokens)
    if absent_query is not None:
        return absent_query
    if len(tokens) == 1 and tokens[0] in _PRONOUN_TARGETS:
        return _normalized_query(
            target=None, negatives=["missing_target"], exists=False
        )

    relation = _extract_relation(tokens)
    if relation is not None and not relation["subject"]:
        return _normalized_query(
            target=None,
            relations=[{"type": relation["type"], "target": relation["target"]}],
            negatives=["missing_target"],
            exists=False,
        )

    action = _extract_action(tokens)
    if action is not None:
        subject_tokens = action["subject"]
        parsed_target = _extract_target(subject_tokens)
        if parsed_target is None:
            return _normalized_query(
                target=None, negatives=["missing_target"], exists=False
            )
        return _normalized_query(
            target=parsed_target["target"],
            attributes=parsed_target["attributes"],
            actions=[{"verb": action["verb"], "target": action["target"]}],
            exists=True,
        )

    if relation is not None:
        parsed_target = _extract_target(relation["subject"])
        if parsed_target is None:
            return _normalized_query(
                target=None, negatives=["missing_target"], exists=False
            )
        return _normalized_query(
            target=parsed_target["target"],
            attributes=parsed_target["attributes"],
            relations=[{"type": relation["type"], "target": relation["target"]}],
            exists=True,
        )

    parsed_target = _extract_target(tokens)
    if parsed_target is None:
        return _normalized_query(
            target=None, negatives=["missing_target"], exists=False
        )
    return _normalized_query(
        target=parsed_target["target"],
        attributes=parsed_target["attributes"],
        exists=True,
    )


def _extract_absent_query(tokens: list[str]) -> NormalizedQuery | None:
    if not tokens or tokens[0] not in _ABSENT_PREFIXES:
        return None
    if not _strip_leading_articles(tokens[1:]):
        return None
    return _normalized_query(target=None, negatives=["absent_object"], exists=False)


def _extract_relation(tokens: list[str]) -> ParsedRelation | None:
    for index, token in enumerate(tokens):
        if token not in _RELATION_WORDS:
            continue
        subject = tokens[:index]
        object_tokens = _strip_leading_articles(tokens[index + 1 :])
        target = " ".join(object_tokens)
        if not target:
            return None
        return {"subject": subject, "type": token, "target": target}
    return None


def _extract_action(tokens: list[str]) -> ParsedAction | None:
    for index, token in enumerate(tokens):
        if token in _ACTION_STANDALONE:
            subject = tokens[:index]
            if not subject:
                return None
            return {"subject": subject, "verb": token, "target": None}
        if token in _ACTION_TARGETS:
            subject = tokens[:index]
            object_tokens = _strip_leading_articles(tokens[index + 1 :])
            target = " ".join(object_tokens) or None
            if not subject:
                return None
            return {"subject": subject, "verb": token, "target": target}
    return None


def _extract_target(tokens: list[str]) -> ParsedTarget | None:
    filtered_tokens = _strip_leading_articles(tokens)
    if not filtered_tokens:
        return None
    target = filtered_tokens[-1]
    if (
        target in _RELATION_WORDS
        or target in _ACTION_STANDALONE
        or target in _ACTION_TARGETS
    ):
        return None
    attributes = [token for token in filtered_tokens[:-1] if token in _ATTRIBUTE_WORDS]
    return {"target": target, "attributes": attributes}


def _strip_leading_articles(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens) and tokens[index] in _LEADING_ARTICLES:
        index += 1
    return tokens[index:]


def _normalized_query(
    *,
    target: str | None,
    attributes: list[str] | None = None,
    relations: list[RelationEntry] | None = None,
    actions: list[ActionEntry] | None = None,
    negatives: list[str] | None = None,
    exists: bool,
) -> NormalizedQuery:
    return {
        "target": target,
        "attributes": attributes or [],
        "relations": relations or [],
        "actions": actions or [],
        "negatives": negatives or [],
        "exists": exists,
    }


def _score_scalar(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _score_lookup(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _mean_constraint_scores(
    keys: list[str], score_lookup: Mapping[str, object]
) -> float:
    if not keys:
        return 1.0
    scores = [_score_scalar(score_lookup.get(key)) for key in keys]
    return sum(scores) / len(scores)


def _relation_key(relation: RelationEntry) -> str:
    return f"{relation['type']}::{relation['target']}"


def _action_key(action: ActionEntry) -> str:
    if action["target"] is None:
        return action["verb"]
    return f"{action['verb']}::{action['target']}"
