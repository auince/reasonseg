#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.BIOtagging.bio_schema import NormalizedQuery
from model.BIOtagging.llm_annotator import AnnotatorConfig, LLMAnnotator, _annotate_one
from model.BIOtagging.reviewed_silver import finalize_reviewed_pairs, tier_counts

DEFAULT_INPUT = ROOT / "model" / "BIOtagging" / "data" / "expanded_silver_mask_complex20k.json"
DEFAULT_OUTPUT = ROOT / "model" / "BIOtagging" / "data" / "deepseek_flash_pro_reviewed_mask20k.json"
DEFAULT_SUMMARY = ROOT / "model" / "BIOtagging" / "data" / "deepseek_flash_pro_reviewed_mask20k_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate mask-generated complex queries with DeepSeek Flash and review with V4-Pro",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--generation-model", default="deepseek-v4-flash")
    parser.add_argument("--review-model", default="deepseek-v4-pro")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--review-batch-size", type=int, default=20)
    parser.add_argument("--fallback-to-seed", action="store_true", default=True)
    return parser


def _load_seed_pairs(path: Path, limit: int) -> list[tuple[str, NormalizedQuery]]:
    data = json.loads(path.read_text())
    pairs = [(item[0], item[1]) for item in data]
    if limit > 0:
        return pairs[:limit]
    return pairs


def _annotate_queries(
    queries: list[str],
    config: AnnotatorConfig,
) -> list[NormalizedQuery | None]:
    results: list[NormalizedQuery | None] = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(_annotate_one, query, config): idx
            for idx, query in enumerate(queries)
        }
        done = 0
        total = len(queries)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  Flash annotated: {done}/{total} ({100 * done / max(total, 1):.0f}%)", flush=True)
    return results


def _review_annotations(
    queries: list[str],
    annotations: list[NormalizedQuery | None],
    config: AnnotatorConfig,
    review_batch_size: int,
) -> list[bool]:
    reviewer = LLMAnnotator(config)
    results: list[bool] = [False] * len(queries)
    valid_items = [(idx, query, ann) for idx, (query, ann) in enumerate(zip(queries, annotations)) if ann is not None]
    for batch_start in range(0, len(valid_items), review_batch_size):
        batch = valid_items[batch_start:batch_start + review_batch_size]
        review_items = [(query, ann) for _, query, ann in batch]
        reviewed = reviewer.review_batch(review_items)
        for (idx, _, _), passed in zip(batch, reviewed):
            results[idx] = passed
        print(
            f"  Pro reviewed: {min(batch_start + review_batch_size, len(valid_items))}/{len(valid_items)}",
            flush=True,
        )
    return results


def main() -> int:
    args = build_parser().parse_args()
    seed_pairs = _load_seed_pairs(args.input, args.limit)
    queries = [query for query, _ in seed_pairs]
    seed_tiers = tier_counts(seed_pairs)
    print(f"Loaded {len(seed_pairs)} seed pairs from {args.input}", flush=True)
    print(f"  Seed tiers: {seed_tiers}", flush=True)

    config = AnnotatorConfig(
        model=args.generation_model,
        review_model=args.review_model,
        workers=args.workers,
    )
    annotations = _annotate_queries(queries, config)
    review_results = _review_annotations(queries, annotations, config, args.review_batch_size)
    finalized = finalize_reviewed_pairs(
        seed_pairs,
        annotations,
        review_results,
        fallback_to_seed=args.fallback_to_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([[q, s] for q, s in finalized], ensure_ascii=False, indent=2))

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "generation_model": args.generation_model,
        "review_model": args.review_model,
        "total_seed_pairs": len(seed_pairs),
        "flash_non_null": sum(1 for ann in annotations if ann is not None),
        "review_passed": sum(1 for ok in review_results if ok),
        "review_failed": sum(1 for ok in review_results if not ok),
        "final_pairs": len(finalized),
        "final_tiers": tier_counts(finalized),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
