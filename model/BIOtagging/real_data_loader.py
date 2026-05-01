from __future__ import annotations

import pickle
from pathlib import Path


def load_refcoco_queries(
    data_root: str | Path,
    *,
    datasets: tuple[str, ...] = ("refcoco", "refcoco+", "refcocog"),
    splits: tuple[str, ...] | None = None,
    max_queries: int | None = None,
) -> dict[str, list[str]]:
    data_root = Path(data_root)

    specs = {
        "refcoco": data_root / "refcoco" / "refs(unc).p",
        "refcoco+": data_root / "refcoco+" / "refs(unc).p",
        "refcocog": data_root / "refcocog" / "refs(umd).p",
    }

    result: dict[str, list[str]] = {}
    for dataset_name in datasets:
        refs_path = specs.get(dataset_name)
        if refs_path is None:
            print(f"  WARNING: unknown dataset {dataset_name}, skipping")
            continue
        if not refs_path.is_file():
            print(f"  WARNING: {refs_path} not found, skipping")
            continue

        with open(refs_path, "rb") as f:
            refs = pickle.load(f)

        filtered_queries: list[str] = []
        seen: set[str] = set()

        for ref in refs:
            if splits is not None and ref.get("split") not in splits:
                continue
            for sent in ref.get("sentences", []):
                q = str(sent.get("raw", "")).strip()
                if q and q not in seen:
                    seen.add(q)
                    filtered_queries.append(q)
                    if max_queries and len(filtered_queries) >= max_queries:
                        break
            if max_queries and len(filtered_queries) >= max_queries:
                break

        result[dataset_name] = filtered_queries
        print(
            f"  {dataset_name}: {len(filtered_queries)} queries "
            f"(from {len(refs)} refs)"
        )

    return result


def export_queries_for_annotation(
    queries_by_dataset: dict[str, list[str]],
    output_dir: str | Path,
    *,
    sample_per_dataset: int | None = None,
) -> tuple[Path, list[str]]:
    import json
    import random

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_queries: list[str] = []
    for dataset_name, queries in queries_by_dataset.items():
        sample = queries
        if sample_per_dataset and len(sample) > sample_per_dataset:
            sample = random.Random(42).sample(sample, sample_per_dataset)
        all_queries.extend(sample)

    output_path = output_dir / "refcoco_queries_for_annotation.json"
    output_path.write_text(
        json.dumps(all_queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Exported {len(all_queries)} queries to {output_path}")
    return output_path, all_queries
