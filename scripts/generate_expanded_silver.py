#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.BIOtagging.bio_schema import structure_to_bio_tags, BIO_TAG_TO_ID
from model.BIOtagging.real_data_loader import load_refcoco_queries
from reasonseg.query import parse_query

DEFAULT_DATA_ROOT = ROOT / "dataset"
DEFAULT_OUTPUT = ROOT / "model" / "BIOtagging" / "data" / "expanded_silver.json"
DEFAULT_MAX_QUERIES = 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate expanded silver BIO data from RefCOCO pickles",
    )
    p.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Root dir containing refcoco/ refcoco+/ refcocog/ subdirs with pickles",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output JSON path for expanded silver data",
    )
    p.add_argument(
        "--max-queries", type=int, default=DEFAULT_MAX_QUERIES,
        help="Cap total queries (0 = unlimited). Applied after dedup across datasets.",
    )
    p.add_argument(
        "--train-splits-only", action="store_true",
        help="Only include queries from train splits (not val/test).",
    )
    return p


def _has_non_o(query_text: str, structure: dict) -> bool:
    tokens = query_text.lower().split()
    tags = structure_to_bio_tags(structure, tokens)
    return any(t != "O" for t in tags)


def main() -> int:
    args = build_parser().parse_args()

    splits = ("train",) if args.train_splits_only else None
    queries_by_dataset = load_refcoco_queries(
        args.data_root, splits=splits,
    )
    total_loaded = sum(len(v) for v in queries_by_dataset.values())
    print(f"Loaded {total_loaded} unique queries across {len(queries_by_dataset)} datasets")

    seen: set[str] = set()
    all_queries: list[str] = []
    for qs in queries_by_dataset.values():
        for q in qs:
            if q not in seen:
                seen.add(q)
                all_queries.append(q)

    print(f"After global dedup: {len(all_queries)} queries")

    if args.max_queries > 0 and len(all_queries) > args.max_queries:
        import random
        rng = random.Random(42)
        all_queries = rng.sample(all_queries, args.max_queries)
        print(f"Capped to {args.max_queries} queries (random sample, seed=42)")

    pairs: list[tuple[str, dict]] = []
    skipped_negative = 0
    skipped_no_labels = 0
    for q in all_queries:
        try:
            struct = parse_query(q)
        except Exception:
            skipped_negative += 1
            continue
        if not struct.get("exists", False):
            skipped_negative += 1
            continue
        if not _has_non_o(q, struct):
            skipped_no_labels += 1
            continue
        pairs.append((q, dict(struct)))

    serializable = [[q, s] for q, s in pairs]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)

    print(f"Wrote {len(pairs)} silver pairs to {args.output}")
    print(f"  Skipped: {skipped_negative} negative/missing-target, "
          f"{skipped_no_labels} all-O labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
