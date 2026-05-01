#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 1 pipeline: extract RefCOCO queries → LLM-annotate → train parser head → review.

Usage:
    # Real data (LLM annotation)
    python model/BIOtagging/pipeline.py --data-root dataset --sample 5000 --llm-model gpt-4o-mini

    # Synthetic data (no LLM needed, instant)
    python model/BIOtagging/pipeline.py --synthetic-count 5000 --skip-llm

    # Mixed: synthetic + rule-parsed real for cold-start
    python model/BIOtagging/pipeline.py --data-root dataset --sample 3000 --synthetic-count 2000 --skip-llm
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.BIOtagging.bio_schema import NormalizedQuery
from model.BIOtagging.config import TrainingConfig
from model.BIOtagging.llm_annotator import (
    AnnotatorConfig,
    batch_annotate_queries,
)
from model.BIOtagging.real_data_loader import (
    export_queries_for_annotation,
    load_refcoco_queries,
)
from model.BIOtagging.review import print_review_report, review_annotations
from model.BIOtagging.synthetic_queries import (
    export_queries_to_file,
    generate_synthetic_queries,
)
from model.BIOtagging.train import BIOTaggingDataset, BIOTrainingRunner

DATA_DIR = Path(__file__).resolve().parent / "data"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1: BIO tagging parser head pipeline")
    p.add_argument("--data-root", type=Path,
                   help="Root containing refcoco/ refcoco+/ refcocog/ subdirs with refs(*).p")
    p.add_argument("--sample", type=int, default=5000,
                   help="Number of real RefCOCO queries to annotate")
    p.add_argument("--synthetic-count", type=int, default=0,
                   help="Additional synthetic queries to mix in")
    p.add_argument("--review-count", type=int, default=200)
    p.add_argument("--train-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--llm-model", default="gpt-4o-mini")
    p.add_argument("--skip-llm", action="store_true",
                   help="Skip LLM annotation (use rule-based parser as silver labels)")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent / "outputs")
    return p


def _query_to_text(query: NormalizedQuery) -> str:
    if not query["exists"]:
        if "absent_object" in query["negatives"] and query["target"]:
            return f"no {query['target']}"
        return "nothing"
    parts: list[str] = list(query["attributes"])
    if query["target"]:
        parts.append(query["target"])
    for rel in query["relations"]:
        parts.extend([rel["type"], rel["target"]])
    for act in query["actions"]:
        parts.append(act["verb"])
        if act["target"]:
            parts.append(act["target"])
    return " ".join(parts)


def main() -> int:
    args = _build_parser().parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    has_real_data = args.data_root is not None and Path(args.data_root).exists()
    has_synthetic = args.synthetic_count > 0

    if not has_real_data and not has_synthetic:
        print("ERROR: provide --data-root or --synthetic-count")
        return 1

    all_gold_queries: list[NormalizedQuery] = []
    all_raw_queries: list[str] = []

    # Step 1a: Load real RefCOCO queries
    real_raw_path = DATA_DIR / "refcoco_queries_for_annotation.json"
    real_raw: list[str] = []
    if has_real_data:
        print(f"Step 1a: Loading real RefCOCO queries from {args.data_root} ...")
        queries_by_ds = load_refcoco_queries(
            args.data_root,
            datasets=("refcoco", "refcoco+", "refcocog"),
            max_queries=args.sample,
        )
        real_raw_path, real_raw = export_queries_for_annotation(
            queries_by_ds, DATA_DIR
        )

    # Step 1b: Synthetic queries
    syn_queries: list[NormalizedQuery] = []
    syn_raw: list[str] = []
    if has_synthetic:
        print(f"Step 1b: Generating {args.synthetic_count} synthetic queries ...")
        syn_queries = generate_synthetic_queries(count=args.synthetic_count)
        syn_raw = [_query_to_text(q) for q in syn_queries]
        export_queries_to_file(syn_queries, DATA_DIR / "synthetic_queries.json")

    # Step 2: LLM annotation on real queries (or rule-based fallback)
    llm_results: list[NormalizedQuery] = []
    if real_raw and not args.skip_llm:
        print(f"Step 2: Annotating {len(real_raw)} real queries with LLM ({args.llm_model}) ...")
        try:
            import openai  # noqa: F401
        except ImportError:
            print("  openai not installed, falling back to rule-based silver labels")
        else:
            config = AnnotatorConfig(model=args.llm_model)
            llm_results = batch_annotate_queries(
                real_raw,
                DATA_DIR / "llm_annotations.json",
                config=config,
                batch_size=20,
            )
            print(f"  {len(llm_results)} queries annotated by LLM")

    # Step 3: Build training data
    print("Step 3: Building training dataset ...")
    if has_synthetic:
        all_gold_queries.extend(syn_queries)
        all_raw_queries.extend(syn_raw)

    if real_raw and llm_results:
        for raw_q, struct in zip(real_raw, llm_results):
            if struct["exists"] or struct["target"]:
                all_gold_queries.append(struct)
                all_raw_queries.append(raw_q)
        print(f"  Added {len(llm_results)} LLM-parsed queries")
    elif real_raw:
        print("  Using rule-based parser as silver labels ...")
        from reasonseg.query import parse_query
        for raw_q in real_raw:
            parsed = parse_query(raw_q)
            all_gold_queries.append(parsed)
            all_raw_queries.append(raw_q)
        print(f"  Rule-parsed {len(real_raw)} queries")

    if not all_gold_queries:
        print("ERROR: No training data")
        return 1

    print(f"  Total: {len(all_gold_queries)} training pairs")
    gold_path = DATA_DIR / "training_gold.json"
    gold_path.write_text(
        json.dumps(all_gold_queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Step 4: Shuffle + split + train
    pairs = list(zip(all_gold_queries, all_raw_queries))
    random.Random(42).shuffle(pairs)
    all_gold_queries[:] = [p[0] for p in pairs]  # type: ignore[index]
    all_raw_queries[:] = [p[1] for p in pairs]  # type: ignore[index]

    val_size = max(1, int(len(all_gold_queries) * 0.1))
    train_ds = BIOTaggingDataset(all_gold_queries[:-val_size], all_raw_queries[:-val_size])
    val_ds = BIOTaggingDataset(all_gold_queries[-val_size:], all_raw_queries[-val_size:])
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    print(f"\nStep 4: Training parser head ({args.train_epochs} epochs) ...")
    trainer_config = TrainingConfig(
        synthetic_queries_count=len(all_gold_queries),
        max_epochs=args.train_epochs,
        batch_size=args.batch_size,
        val_split_ratio=0.0,
        output_dir=args.output_dir,
    )
    runner = BIOTrainingRunner(trainer_config)
    metrics = runner.train(train_ds, val_ds)
    print(f"  Best val accuracy: {metrics['best_val_acc']:.4f}")
    print(f"  Model saved to: {args.output_dir / trainer_config.model_save_name}")

    # Step 5: Review (only when LLM was used)
    if llm_results:
        print(f"\nStep 5: Reviewing LLM annotations (sample={min(args.review_count, len(real_raw))}) ...")
        review_data = []
        for idx in range(min(args.review_count, len(real_raw))):
            llm_s = llm_results[idx] if idx < len(llm_results) else None
            review_data.append({
                "raw_query": real_raw[idx],
                "gold_structure": all_gold_queries[idx] if idx < len(all_gold_queries) else None,
                "predicted_structure": llm_s,
            })
        review_path = DATA_DIR / "review_data.json"
        review_path.write_text(json.dumps(review_data, ensure_ascii=False, indent=2))
        entries = review_annotations(review_path, sample_count=min(args.review_count, len(review_data)))
        print_review_report(entries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
