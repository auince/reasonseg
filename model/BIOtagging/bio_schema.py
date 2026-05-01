from __future__ import annotations

from typing import TypedDict

# ── BIO tag vocabulary (13 tags, matching learnable_query_parser.md) ──────────
BIO_TAGS: tuple[str, ...] = (
    "O",
    "B-TGT", "I-TGT",
    "B-ATTR", "I-ATTR",
    "B-REL", "I-REL",
    "B-REL-TGT", "I-REL-TGT",
    "B-ACT", "I-ACT",
    "B-ACT-TGT", "I-ACT-TGT",
    "B-NEG",
)

BIO_TAG_TO_ID: dict[str, int] = {tag: idx for idx, tag in enumerate(BIO_TAGS)}
ID_TO_BIO_TAG: dict[int, str] = {idx: tag for idx, tag in enumerate(BIO_TAGS)}


# ── TypedDict schema (mirrors reasonseg/query.py) ────────────────────────────

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


# ── BIO → Structure (deterministic reverse converter) ────────────────────────

def bio_tags_to_structure(tokens: list[str], tags: list[str]) -> NormalizedQuery:
    if not tokens or len(tokens) != len(tags):
        return _empty_query(["malformed_query"])

    # Detect negation
    negatives: list[str] = []
    for tok, tag in zip(tokens, tags):
        if tag == "B-NEG":
            negatives.append("absent_object")

    def _extract_spans(prefix: str) -> list[str]:
        spans: list[list[str]] = []
        current: list[str] = []
        for tok, tag in zip(tokens, tags):
            if tag == f"B-{prefix}":
                if current:
                    spans.append(current)
                current = [tok]
            elif tag == f"I-{prefix}":
                current.append(tok)
            elif tag.startswith("B-") and current:
                spans.append(current)
                current = []
            elif tag == "O" and current:
                spans.append(current)
                current = []
        if current:
            spans.append(current)
        return [" ".join(span) for span in spans]

    targets = _extract_spans("TGT")
    attributes = _extract_spans("ATTR")
    rel_targets = _extract_spans("REL-TGT")
    act_targets = _extract_spans("ACT-TGT")

    # Extract relation words (B-REL / I-REL)
    rel_words = _extract_spans("REL")
    rel_words = [w for w in rel_words if not w.startswith("REL-TGT")]

    # Extract action words (B-ACT / I-ACT)
    act_words = _extract_spans("ACT")
    act_words = [w for w in act_words if not w.startswith("ACT-TGT")]

    if negatives and negatives[0] == "absent_object":
        return _make_query(
            target=None,
            attributes=[],
            relations=_pair_relations(rel_words, rel_targets),
            actions=_pair_actions(act_words, act_targets),
            negatives=negatives,
            exists=False,
        )

    target = targets[0] if targets else None
    if target is None and not attributes and not rel_words and not act_words:
        return _empty_query(["missing_target"])

    return _make_query(
        target=target,
        attributes=attributes,
        relations=_pair_relations(rel_words, rel_targets),
        actions=_pair_actions(act_words, act_targets),
        negatives=[],
        exists=True,
    )


def _pair_relations(rel_words: list[str], rel_targets: list[str]) -> list[RelationEntry]:
    relations: list[RelationEntry] = []
    for i, rel in enumerate(rel_words):
        tgt = rel_targets[i] if i < len(rel_targets) else rel_words[i] if i + 1 < len(rel_words) else ""
        relations.append({"type": rel, "target": tgt})
    return relations


def _pair_actions(act_words: list[str], act_targets: list[str]) -> list[ActionEntry]:
    actions: list[ActionEntry] = []
    for i, verb in enumerate(act_words):
        tgt = act_targets[i] if i < len(act_targets) else None
        actions.append({"verb": verb, "target": tgt})
    return actions


def _empty_query(negatives: list[str]) -> NormalizedQuery:
    return _make_query(target=None, attributes=[], relations=[], actions=[], negatives=negatives, exists=False)


def _make_query(
    *,
    target: str | None,
    attributes: list[str],
    relations: list[RelationEntry],
    actions: list[ActionEntry],
    negatives: list[str],
    exists: bool,
) -> NormalizedQuery:
    return {
        "target": target,
        "attributes": attributes,
        "relations": relations,
        "actions": actions,
        "negatives": negatives,
        "exists": exists,
    }


# ── Structure → BIO (deterministic forward converter) ────────────────────────

def structure_to_bio_tags(query: NormalizedQuery, tokens: list[str]) -> list[str]:
    n = len(tokens)
    tags = ["O"] * n

    if not query["exists"]:
        if "absent_object" in query["negatives"] and n > 0:
            tags[0] = "B-NEG"
        return tags

    # Tag target
    if query["target"]:
        _tag_entity(tokens, tags, query["target"], "TGT")

    # Tag attributes
    for attr in query["attributes"]:
        _tag_entity(tokens, tags, attr, "ATTR")

    # Tag relations
    for rel in query["relations"]:
        rel_type = rel["type"]
        _tag_entity(tokens, tags, rel_type, "REL")
        if rel["target"]:
            _tag_entity(tokens, tags, rel["target"], "REL-TGT")

    # Tag actions
    for act in query["actions"]:
        verb = act["verb"]
        _tag_entity(tokens, tags, verb, "ACT")
        if act["target"]:
            _tag_entity(tokens, tags, act["target"], "ACT-TGT")

    return tags


def _tag_entity(tokens: list[str], tags: list[str], entity: str, prefix: str) -> None:
    entity_tokens = entity.lower().split()
    for start in range(len(tokens) - len(entity_tokens) + 1):
        if tokens[start : start + len(entity_tokens)] == entity_tokens:
            tags[start] = f"B-{prefix}"
            for j in range(1, len(entity_tokens)):
                tags[start + j] = f"I-{prefix}"
            return


# ── Convenience: direct tokens → label ids ────────────────────────────────────

def tokens_to_bio_labels(
    tokens: list[str], query_struct: NormalizedQuery
) -> list[int]:
    tags = structure_to_bio_tags(query_struct, tokens)
    return [BIO_TAG_TO_ID[tag] for tag in tags]
