#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.BIOtagging.reviewed_silver import merge_silver_pairs, tier_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge base and reviewed silver pairs")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_pairs(path: Path) -> list[tuple[str, dict]]:
    data = json.loads(path.read_text())
    return [(item[0], item[1]) for item in data]


def main() -> int:
    args = build_parser().parse_args()
    base_pairs = _load_pairs(args.base)
    overlay_pairs = _load_pairs(args.overlay)
    merged = merge_silver_pairs(base_pairs, overlay_pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([[q, s] for q, s in merged], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "base": len(base_pairs),
                "overlay": len(overlay_pairs),
                "merged": len(merged),
                "tiers": tier_counts(merged),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
