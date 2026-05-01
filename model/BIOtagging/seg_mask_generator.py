"""Silver data generator: synthesize [query_text, NormalizedQuery] pairs
from the seg_mask_per_instance.json mask/bbox/metadata file.

The JSON has keys like ``{image_id}-{instance_idx}-{category}.jpg`` and
values containing ``segmentation``, ``bbox``, and ``image_dim``.  No text
fields are present, so query texts and structures are synthesized from:

- **category name** → target noun
- **bbox size/position** → size & position adjectives (attributes)
- **pairwise bbox centres** → spatial relations between same-image instances

Generated pairs are scored with ``complexity_selection.score_complexity``
and the most complex are selected via ``select_top_complex``.

Output is a JSON list ``[[query_text, NormalizedQuery], ...]`` consumable
by ``scripts/train_parser_head_stage1_fast.py --silver-path ...``.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bio_schema import NormalizedQuery
from .complexity_selection import score_complexity, select_top_complex

_AREA_TINY = 0.02
_AREA_SMALL = 0.05
_AREA_LARGE = 0.30
_AREA_HUGE = 0.50
_ASPECT_TALL = 1.5
_ASPECT_WIDE = 1.5

_POS_HORIZ = ("left", "center", "right")
_POS_VERT = ("top", "middle", "bottom")


@dataclass(frozen=True)
class _Instance:
    image_id: str
    instance_idx: int
    category: str
    bbox: list[float]
    image_dim: list[int]
    key: str

    @property
    def target(self) -> str:
        return self.category


def _parse_key(key: str) -> _Instance:
    stem = key.rsplit(".jpg", 1)[0]
    parts = stem.split("-")
    numeric_indices = [i for i, p in enumerate(parts) if p.isdigit()]
    if len(numeric_indices) < 2:
        raise ValueError(f"Unexpected key format: {key!r}")
    image_id = parts[numeric_indices[0]]
    instance_idx = int(parts[numeric_indices[1]])
    category = "-".join(parts[numeric_indices[1] + 1 :])
    return _Instance(
        image_id=image_id,
        instance_idx=instance_idx,
        category=_category_clean(category),
        bbox=[],
        image_dim=[],
        key=key,
    )


def _category_clean(raw: str) -> str:
    return raw.lower().replace("-", " ").replace("_", " ")


def _compute_centre(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _area_fraction(bbox: list[float], image_dim: list[int]) -> float:
    _, _, w, h = bbox
    ih, iw = image_dim
    return (w * h) / (iw * ih)


def _aspect_ratio(bbox: list[float]) -> float:
    _, _, w, h = bbox
    if w < 1e-6:
        return 999.0
    return h / w


def _generate_size_attrs(bbox: list[float], image_dim: list[int]) -> list[str]:
    attrs: list[str] = []
    area = _area_fraction(bbox, image_dim)
    aspect = _aspect_ratio(bbox)

    if area < _AREA_TINY:
        attrs.append("tiny")
    elif area < _AREA_SMALL:
        attrs.append("small")
    elif area > _AREA_HUGE:
        attrs.append("huge")
    elif area > _AREA_LARGE:
        attrs.append("large")

    if aspect > _ASPECT_TALL:
        attrs.append("tall")
    elif 1.0 / aspect > _ASPECT_WIDE:
        attrs.append("wide")

    return attrs


def _generate_position_attrs(bbox: list[float], image_dim: list[int]) -> list[str]:
    cx, cy = _compute_centre(bbox)
    ih, iw = image_dim

    h_bin = int(cx / iw * 3)
    v_bin = int(cy / ih * 3)
    h_label = _POS_HORIZ[min(h_bin, 2)]
    v_label = _POS_VERT[min(v_bin, 2)]

    if h_label == "center" and v_label == "middle":
        return []
    if h_label == "center":
        return [v_label]
    if v_label == "middle":
        return [h_label]
    return [f"{v_label}-{h_label}"]


def _spatial_relation(
    subj: _Instance, obj: _Instance,
) -> tuple[str, str] | None:
    if subj.image_id != obj.image_id or subj.key == obj.key:
        return None

    cx1, cy1 = _compute_centre(subj.bbox)
    cx2, cy2 = _compute_centre(obj.bbox)

    dx = cx2 - cx1
    dy = cy2 - cy1
    ih, iw = subj.image_dim

    if abs(dx) < 0.03 * iw and abs(dy) < 0.03 * ih:
        return None
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    if abs_dx >= abs_dy * 1.5:
        rel = "right of" if dx > 0 else "left of"
    elif abs_dy >= abs_dx * 1.5:
        rel = "below" if dy > 0 else "above"
    else:
        rel = _diagonal_relation(dx, dy)

    return rel, obj.target


def _diagonal_relation(dx: float, dy: float) -> str:
    h = "right of" if dx > 0 else "left of"
    v = "below" if dy > 0 else "above"
    return f"{v} and {h}"


def _make_noun_query(inst: _Instance) -> tuple[str, NormalizedQuery] | None:
    tgt = inst.target
    if not tgt:
        return None
    return tgt, _q(target=tgt)


def _make_attr_query(inst: _Instance) -> tuple[str, NormalizedQuery] | None:
    tgt = inst.target
    if not tgt:
        return None
    attrs = _generate_size_attrs(inst.bbox, inst.image_dim)
    pos = _generate_position_attrs(inst.bbox, inst.image_dim)
    all_attrs = attrs + pos
    if not all_attrs:
        return None
    query = " ".join(all_attrs + [tgt])
    return query, _q(target=tgt, attributes=all_attrs)


def _make_rel_query(
    subj: _Instance, obj: _Instance,
) -> tuple[str, NormalizedQuery] | None:
    rel_info = _spatial_relation(subj, obj)
    if rel_info is None:
        return None
    rel_word, rel_tgt = rel_info
    query = f"{subj.target} {rel_word} {rel_tgt}"
    return query, _q(
        target=subj.target,
        relations=[{"type": rel_word, "target": rel_tgt}],
    )


def _make_rel_attr_query(
    subj: _Instance, obj: _Instance,
) -> tuple[str, NormalizedQuery] | None:
    rel_info = _spatial_relation(subj, obj)
    if rel_info is None:
        return None
    rel_word, rel_tgt = rel_info
    attrs = _generate_size_attrs(subj.bbox, subj.image_dim)
    if not attrs:
        return None
    query = " ".join(attrs + [subj.target, rel_word, rel_tgt])
    return query, _q(
        target=subj.target,
        attributes=attrs,
        relations=[{"type": rel_word, "target": rel_tgt}],
    )


def _q(
    target: str | None = None,
    attributes: list[str] | None = None,
    relations: list[dict[str, str]] | None = None,
    actions: list[dict[str, str | None]] | None = None,
    negatives: list[str] | None = None,
    exists: bool = True,
) -> NormalizedQuery:
    return {
        "target": target,
        "attributes": attributes or [],
        "relations": relations or [],
        "actions": actions or [],
        "negatives": negatives or [],
        "exists": exists,
    }


def generate_pairs_from_seg_mask(
    json_path: Path, seed: int = 7, max_noun: int = 0
) -> list[tuple[str, NormalizedQuery]]:
    """Produce all synthesizable (query_text, NormalizedQuery) pairs.

    Parameters
    ----------
    json_path : Path
        Path to ``seg_mask_per_instance.json``.
    seed : int
        Random seed for deterministic shuffling of spatial-relation sampling.
    max_noun : int
        Cap on simple noun-only queries.  0 means unlimited.  These are
        low-complexity; set a limit if you want to favour higher-complexity
        samples in the final selection.

    Returns
    -------
    list of (str, NormalizedQuery) pairs.
    """
    raw: dict[str, Any] = json.loads(json_path.read_text())
    rng = random.Random(seed)

    by_image: dict[str, list[_Instance]] = defaultdict(list)
    for key, val in raw.items():
        inst = _parse_key(key)
        by_image[inst.image_id].append(
            _Instance(
                image_id=inst.image_id,
                instance_idx=inst.instance_idx,
                category=inst.category,
                bbox=val["bbox"],
                image_dim=val["image_dim"],
                key=key,
            )
        )

    pairs: list[tuple[str, NormalizedQuery]] = []
    noun_count = 0

    for image_id, instances in by_image.items():
        for inst in instances:
            if max_noun == 0 or noun_count < max_noun:
                noun = _make_noun_query(inst)
                if noun:
                    pairs.append(noun)
                    noun_count += 1

            attr = _make_attr_query(inst)
            if attr:
                pairs.append(attr)

        if 2 <= len(instances) <= 20:
            for i in range(len(instances)):
                candidates = [instances[j] for j in range(len(instances)) if j != i]
                rng.shuffle(candidates)
                for obj in candidates[:3]:
                    rel = _make_rel_query(instances[i], obj)
                    if rel:
                        pairs.append(rel)

                    rel_attr = _make_rel_attr_query(instances[i], obj)
                    if rel_attr:
                        pairs.append(rel_attr)

    seen: set[str] = set()
    dedup: list[tuple[str, NormalizedQuery]] = []
    for q, s in pairs:
        if q not in seen:
            seen.add(q)
            dedup.append((q, s))
    return dedup


def export_expanded_silver(
    pairs: list[tuple[str, NormalizedQuery]],
    output_path: Path,
    top_k: int | None = None,
) -> Path:
    """Select top-k most complex pairs and write as JSON consumable by
    ``train_parser_head_stage1_fast.py --silver-path ...``.

    The output format is ``[[query_text, structure], ...]``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if top_k is not None and top_k > 0:
        pairs = select_top_complex(pairs, top_k)
    data: list[list[object]] = [[q, s] for q, s in pairs]
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def generate_expanded_silver(
    seg_mask_path: Path,
    output_path: Path,
    *,
    seed: int = 7,
    max_noun: int = 0,
    top_k: int | None = None,
) -> Path:
    """Convenience: generate pairs, optionally select top-k, and export."""
    pairs = generate_pairs_from_seg_mask(seg_mask_path, seed=seed, max_noun=max_noun)
    return export_expanded_silver(pairs, output_path, top_k=top_k)
