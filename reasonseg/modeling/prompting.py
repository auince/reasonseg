from __future__ import annotations

from typing import Any


def infer_requested_target(raw_prompt: str, query_struct: dict[str, Any]) -> str:
    target = query_struct.get("target")
    if isinstance(target, str) and target:
        return target

    raw_tokens = raw_prompt.strip().lower().split()
    if raw_tokens and raw_tokens[0] in {"no", "without", "absent"}:
        raw_tokens = raw_tokens[1:]
    while raw_tokens and raw_tokens[0] in {"the", "a", "an"}:
        raw_tokens = raw_tokens[1:]
    requested_target = " ".join(raw_tokens).strip()
    return requested_target or raw_prompt.strip().lower()


def infer_slice_tag(query_struct: dict[str, Any]) -> str:
    if query_struct["exists"] is False:
        return "no_target"
    if query_struct["relations"] or query_struct["actions"]:
        return "relation_action"
    if query_struct["attributes"]:
        return "attribute"
    return "noun"


def compose_reasonseg_prompt(
    query_struct: dict[str, Any],
    fallback_text: str,
) -> str:
    target = query_struct.get("target")
    if not isinstance(target, str) or not target:
        return fallback_text

    prompt_parts = list(query_struct.get("attributes", []))
    prompt_parts.append(target)

    for relation in query_struct.get("relations", []):
        relation_type = relation.get("type")
        relation_target = relation.get("target")
        if isinstance(relation_type, str) and isinstance(relation_target, str):
            prompt_parts.extend([relation_type, relation_target])

    for action in query_struct.get("actions", []):
        action_verb = action.get("verb")
        action_target = action.get("target")
        if isinstance(action_verb, str):
            prompt_parts.append(action_verb)
            if isinstance(action_target, str) and action_target:
                prompt_parts.append(action_target)

    return " ".join(prompt_parts)
