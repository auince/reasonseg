# AGENTS.md — ReasonSeg

## Environment

- Use conda env `reasonseg-py311`.
- Run repo commands from `/home/lch/Project/ReasonSeg`.
- Real dataset root is `/home/lch/Project/ReasonSeg/datasets`. Ignore older `/data/lch/...` examples in `TRAINING_LAUNCH.md`; they are historical.

## Highest-value command order

1. Verify or materialize data first:
   - `conda run -n reasonseg-py311 python scripts/data/prepare_refcoco.py --data-root /home/lch/Project/ReasonSeg/datasets --verify-only`
   - `conda run -n reasonseg-py311 python scripts/data/prepare_refcoco.py --data-root /home/lch/Project/ReasonSeg/datasets --materialize`
2. Train.
3. Eval or test with an explicit `--checkpoint`.
4. Benchmark only after collecting prediction JSONs into a clean prediction root.

## Canonical commands

- Preferred training launcher:
  - `conda run -n reasonseg-py311 python scripts/yolo_train.py --task refcoco --data-root /home/lch/Project/ReasonSeg/datasets --project /home/lch/Project/ReasonSeg/outputs --name my_run --device 0,1 --batch 2 --lr 0.0002 --max-iter 90000`
- Low-level train:
  - `conda run -n reasonseg-py311 python scripts/train.py --config configs/refcoco/refcoco_reasonseg.yaml --data-root /home/lch/Project/ReasonSeg/datasets --output-dir /tmp/reasonseg-train --max-iter 1`
- Eval:
  - `conda run -n reasonseg-py311 python scripts/eval.py --config configs/refcoco/refcoco_reasonseg.yaml --data-root /home/lch/Project/ReasonSeg/datasets --checkpoint /path/to/model_final.pth --split refcoco_val_unc --output-dir /tmp/reasonseg-eval`
- Test:
  - `conda run -n reasonseg-py311 python scripts/test.py --config configs/refcoco/refcoco_plus_reasonseg.yaml --data-root /home/lch/Project/ReasonSeg/datasets --checkpoint /path/to/model_final.pth --split refcoco_plus_testA_unc --output-dir /tmp/reasonseg-test`
- Benchmark export:
  - `conda run -n reasonseg-py311 python scripts/benchmark/run_benchmark.py --spec benchmarks/refexp_paper_benchmark.json --pred-root /path/to/preds --output /tmp/reasonseg-benchmark.json`
- Full tests:
  - `pytest tests/`
- Focused test:
  - `pytest tests/reasonseg/test_query_parser.py`

## Repo shape

- `reasonseg/cli_surface.py` is the single source of truth for CLI args for `train.py`, `eval.py`, `test.py`, `scripts/data/prepare_refcoco.py`, and `scripts/benchmark/run_benchmark.py`.
- `scripts/yolo_train.py` is the high-level entrypoint. Chain: `yolo_train.py` → `watch_train.py` → `accelerate launch` (multi-GPU only) → `scripts/train.py`.
- `reasonseg/runtime/train.py` and `reasonseg/runtime/eval.py` own the real loops. This is not Detectron2's default trainer flow even though config style looks Detectron2-like.
- `configs/refcoco/refcoco_base.yaml` is the base YACS config; child configs only swap dataset settings or enable ReasonSeg mode.
- `reasonseg/query.py` is still a small rule-based parser with a tiny hardcoded vocabulary; do not assume broad natural-language coverage.
- `model/segment_anything_2/` and `model/unilm/` are vendored forks. Avoid refactoring them unless the task explicitly targets backend fork code.

## Verification facts

- There is no `pyproject.toml`, no `pytest.ini`, and no pre-commit config in the repo root.
- Pytest shared fixtures live in `tests/reasonseg/conftest.py`.
- Pyright is configured by `pyrightconfig.json` for `reasonseg`, `scripts`, `tests`, and top-level `model`, but excludes `model/segment_anything_2/**` and `model/unilm/**`.
- `black` is pinned in `requirements_reasonseg.txt`, but there is no formatter config to infer style from.

## Operational gotchas

- `scripts/data/prepare_refcoco.py` needs at least one mode flag: `--verify-only`, `--materialize`, or both. Calling it with neither is an error.
- `--data-root` must contain the raw RefCOCO-family assets plus COCO images/annotations. Materialization verifies this layout.
- Eval and test never resume from `last_checkpoint`; `--checkpoint` is always required.
- In `scripts/yolo_train.py`, `--resume` and `--checkpoint` are mutually exclusive.
- Multi-GPU batch size is global `SOLVER.IMS_PER_BATCH`, not per-GPU. On `--device 0,1`, `--batch` must be divisible by 2; `--batch 1` is invalid.
- `accelerate_config.yaml` is currently set for 2 local GPUs (`gpu_ids: 0,1`, `num_processes: 2`) with `mixed_precision: 'no'`.
- Dual-GPU runs are prone to CUDA OOM; the watchdog exists because training can stall or go idle under failure.
- Training writes outputs under `<output-dir>/run_0/`; eval/test write predictions and metrics under `<output-dir>/run_0/inference/`.
- The benchmark runner scans `--pred-root` recursively. Keep that tree clean or scoring can fail or include unintended files.
- Supported datasets are RefCOCO UNC, RefCOCO+ UNC, and RefCOCOg UMD. RefCOCOg Google splits are not supported.

## Style cues visible in code

- New Python files should use `from __future__ import annotations`.
- Prefer `pathlib.Path` over string path manipulation.
