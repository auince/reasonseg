#!/usr/bin/env python3
from __future__ import annotations
"""Annotate RefCOCO queries: deepseek-chat (16 workers) + v4-pro review every 100."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.BIOtagging.real_data_loader import load_refcoco_queries
from model.BIOtagging.llm_annotator import AnnotatorConfig, batch_annotate_queries

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

N = 3000
per_ds = N // 3

qmap = load_refcoco_queries(
    "/home/lch/Project/ReasonSeg/dataset",
    datasets=("refcoco", "refcoco+", "refcocog"),
    max_queries=per_ds,
)

all_q = []
for ds, qs in qmap.items():
    all_q.extend(qs)

total = len(all_q)
print(f"Loaded {total} queries")
for ds, qs in qmap.items():
    print(f"  {ds}: {len(qs)}")

cfg = AnnotatorConfig(
    model="deepseek-chat",
    review_model="deepseek-v4-pro",
    workers=16,
    review_interval=100,
    review_sample=10,
)

out = DATA / "llm_annotations_3k.json"
results = batch_annotate_queries(all_q, out, config=cfg, batch_size=256)

import json
out_summary = DATA / "llm_annotations_summary.json"
json.dump({
    "total_loaded": total,
    "total_annotated": len(results),
    "per_dataset": {ds: len(qs) for ds, qs in qmap.items()},
}, out_summary, ensure_ascii=False, indent=2)

print(f"\nDONE: {len(results)} annotations saved to {out}")
