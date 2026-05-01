from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .bio_schema import NormalizedQuery, structure_to_bio_tags


@dataclass
class ReviewEntry:
    index: int
    raw_query: str
    gold_structure: NormalizedQuery
    predicted_structure: NormalizedQuery | None
    tokens: list[str]
    gold_tags: list[str]
    predicted_tags: list[str] | None
    is_match: bool
    errors: list[str]


def review_annotations(
    annotations_path: Path,
    sample_count: int = 200,
    random_seed: int = 42,
) -> list[ReviewEntry]:
    import random
    random.seed(random_seed)

    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    indices = list(range(len(data)))
    if len(indices) > sample_count:
        indices = random.sample(indices, sample_count)

    entries: list[ReviewEntry] = []
    for idx in indices:
        item = data[idx]
        raw_query = item.get("raw_query", "")
        gold = _normalize_structure(item.get("gold_structure", {}))

        tokens = raw_query.lower().split()
        gold_tags = structure_to_bio_tags(gold, tokens)

        pred = item.get("predicted_structure")
        if pred is not None:
            pred = _normalize_structure(pred)
            pred_tags = structure_to_bio_tags(pred, tokens)
            is_match = _structures_equal(gold, pred)
            errors = _diff_structures(gold, pred)
        else:
            pred_tags = None
            is_match = False
            errors = ["no_prediction"]

        entries.append(
            ReviewEntry(
                index=idx,
                raw_query=raw_query,
                gold_structure=gold,
                predicted_structure=pred,
                tokens=tokens,
                gold_tags=gold_tags,
                predicted_tags=pred_tags,
                is_match=is_match,
                errors=errors,
            )
        )

    return entries


def print_review_report(entries: list[ReviewEntry]) -> None:
    total = len(entries)
    matches = sum(1 for e in entries if e.is_match)
    failure = [e for e in entries if not e.is_match]

    print(f"Review Report: {matches}/{total} matched ({100*matches/total:.1f}%)")
    print(f"Noun queries: {_count_by_type(entries, 'noun')}")
    print(f"Attribute queries: {_count_by_type(entries, 'attribute')}")
    print(f"Relation queries: {_count_by_type(entries, 'relation')}")
    print(f"Action queries: {_count_by_type(entries, 'action')}")
    print(f"Absent queries: {_count_by_type(entries, 'absent')}")
    print(f"Match rate by type:")
    for qtype in ["noun", "attribute", "relation", "action", "absent"]:
        typed = [e for e in entries if _classify_query(e.gold_structure) == qtype]
        if typed:
            acc = sum(1 for e in typed if e.is_match) / len(typed)
            print(f"  {qtype}: {acc:.1%} ({len(typed)} samples)")

    if failure:
        print(f"\nFailure examples (showing first 5):")
        for entry in failure[:5]:
            print(f"  [{entry.index}] {entry.raw_query!r}")
            print(f"    Errors: {entry.errors}")
            if entry.predicted_structure:
                print(f"    Pred: {_structure_summary(entry.predicted_structure)}")


def _normalize_structure(raw: dict) -> NormalizedQuery:
    return {
        "target": raw.get("target"),
        "attributes": raw.get("attributes") if isinstance(raw.get("attributes"), list) else [],
        "relations": _normalize_rel_list(raw.get("relations", [])),
        "actions": _normalize_act_list(raw.get("actions", [])),
        "negatives": raw.get("negatives") if isinstance(raw.get("negatives"), list) else [],
        "exists": bool(raw.get("exists", True)),
    }


def _normalize_rel_list(raw: list) -> list[dict]:
    result = []
    for item in raw:
        if isinstance(item, dict):
            result.append({
                "type": str(item.get("type", "")),
                "target": str(item.get("target", "")),
            })
    return result


def _normalize_act_list(raw: list) -> list[dict]:
    result = []
    for item in raw:
        if isinstance(item, dict):
            tgt = item.get("target")
            result.append({
                "verb": str(item.get("verb", "")),
                "target": str(tgt) if tgt else None,
            })
    return result


def _structures_equal(a: NormalizedQuery, b: NormalizedQuery) -> bool:
    return (
        a["target"] == b["target"]
        and sorted(a["attributes"]) == sorted(b["attributes"])
        and len(a["relations"]) == len(b["relations"])
        and all(r in b["relations"] for r in a["relations"])
        and len(a["actions"]) == len(b["actions"])
        and all(ac in b["actions"] for ac in a["actions"])
        and a["exists"] == b["exists"]
    )


def _diff_structures(a: NormalizedQuery, b: NormalizedQuery) -> list[str]:
    errors = []
    if a["target"] != b["target"]:
        errors.append(f"target: {a['target']} != {b['target']}")
    if sorted(a["attributes"]) != sorted(b["attributes"]):
        errors.append(f"attrs: {a['attributes']} != {b['attributes']}")
    if a["relations"] != b["relations"]:
        errors.append(f"rels: {a['relations']} != {b['relations']}")
    if a["actions"] != b["actions"]:
        errors.append(f"acts: {a['actions']} != {b['actions']}")
    if a["exists"] != b["exists"]:
        errors.append(f"exists: {a['exists']} != {b['exists']}")
    return errors


def _classify_query(q: NormalizedQuery) -> str:
    if not q["exists"]:
        return "absent" if "absent_object" in q["negatives"] else "absent"
    if q["actions"]:
        return "action"
    if q["relations"]:
        return "relation"
    if q["attributes"]:
        return "attribute"
    return "noun"


def _structure_summary(q: NormalizedQuery) -> str:
    parts = []
    if q["target"]:
        parts.append(f"tgt={q['target']}")
    if q["attributes"]:
        parts.append(f"attr={q['attributes']}")
    if q["relations"]:
        parts.append(f"rel={q['relations']}")
    if q["actions"]:
        parts.append(f"act={q['actions']}")
    if not q["exists"]:
        parts.append("exists=false")
    return ", ".join(parts)


def _count_by_type(entries: list[ReviewEntry], qtype: str) -> int:
    return sum(1 for e in entries if _classify_query(e.gold_structure) == qtype)
