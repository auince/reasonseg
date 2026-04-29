# ReasonSeg RefCOCO Paper Experiment Handoff

This repository owns the full paper-facing RefCOCO workflow from local data checks to train, eval, test, and benchmark export.

## Verified environment

- Repo root: `/home/lch/Project/ReasonSeg`
- Verified conda env: `reasonseg-py311`
- Local dataset root: `/home/lch/Project/ReasonSeg/datasets`
- Supported paper dataset families: RefCOCO UNC, RefCOCO+ UNC, RefCOCOg UMD

All commands below were run from the repo root with `conda run -n reasonseg-py311`.

## Dataset preparation and verification

Check that the local RefCOCO-family assets and the materialized runtime JSON files are ready:

```bash
conda run -n reasonseg-py311 python scripts/data/prepare_refcoco.py --data-root /home/lch/Project/ReasonSeg/datasets --verify-only
```

If you need to rebuild the materialized runtime JSON files from the local raw assets under `datasets/refcoco`, `datasets/refcoco+`, and `datasets/refcocog`, use:

```bash
conda run -n reasonseg-py311 python scripts/data/prepare_refcoco.py --data-root /home/lch/Project/ReasonSeg/datasets --materialize
```

## Config and split matrix

- `configs/refcoco/refcoco_reasonseg.yaml`
  - train: `refcoco_train_unc`
  - eval: `refcoco_val_unc`
  - test: `refcoco_testA_unc`, `refcoco_testB_unc`
- `configs/refcoco/refcoco_plus_reasonseg.yaml`
  - train: `refcoco_plus_train_unc`
  - eval: `refcoco_plus_val_unc`
  - test: `refcoco_plus_testA_unc`, `refcoco_plus_testB_unc`
- `configs/refcoco/refcocog_reasonseg.yaml`
  - train: `refcocog_train_umd`
  - eval: `refcocog_val_umd`
  - test: `refcocog_test_umd`

RefCOCOg Google splits are not part of this repo contract.

## Verified repo-root command surface

CLI audit:

```bash
conda run -n reasonseg-py311 python scripts/train.py --help
conda run -n reasonseg-py311 python scripts/eval.py --help
conda run -n reasonseg-py311 python scripts/test.py --help
conda run -n reasonseg-py311 python scripts/data/prepare_refcoco.py --help
conda run -n reasonseg-py311 python scripts/benchmark/run_benchmark.py --help
```

Verified one-iteration training smoke:

```bash
conda run -n reasonseg-py311 python scripts/train.py --config configs/refcoco/refcoco_reasonseg.yaml --data-root /home/lch/Project/ReasonSeg/datasets --output-dir /home/lch/Project/ReasonSeg/.sisyphus/evidence/task-10-train-smoke --max-iter 1
```

Verified RefCOCO validation eval:

```bash
conda run -n reasonseg-py311 python scripts/eval.py --config configs/refcoco/refcoco_reasonseg.yaml --data-root /home/lch/Project/ReasonSeg/datasets --checkpoint /home/lch/Project/ReasonSeg/.sisyphus/evidence/task-10-train-smoke/run_0/model_final.pth --split refcoco_val_unc --output-dir /home/lch/Project/ReasonSeg/.sisyphus/evidence/task-10-eval-refcoco-val
```

Verified RefCOCO+ testA run:

```bash
conda run -n reasonseg-py311 python scripts/test.py --config configs/refcoco/refcoco_plus_reasonseg.yaml --data-root /home/lch/Project/ReasonSeg/datasets --checkpoint /home/lch/Project/ReasonSeg/.sisyphus/evidence/task-10-train-smoke/run_0/model_final.pth --split refcoco_plus_testA_unc --output-dir /home/lch/Project/ReasonSeg/.sisyphus/evidence/task-10-test-refcocoplus-testa
```

For your own paper runs, keep the same command shape and swap only the config, split, checkpoint, and output paths you want to own.

## Canonical run artifacts

Every command writes into `<output-dir>/run_<idx>/`.

- always written
  - `<output-dir>/run_<idx>/config.yaml`
  - `<output-dir>/run_<idx>/log.txt`
- training writes
  - `<output-dir>/run_<idx>/train_metrics.json`
  - `<output-dir>/run_<idx>/model_final.pth`
  - `<output-dir>/run_<idx>/last_checkpoint`
- eval and test write
  - `<output-dir>/run_<idx>/inference/predictions.json`
  - `<output-dir>/run_<idx>/inference/metrics.json`

`last_checkpoint` is only a training resume pointer. Eval and test always require an explicit `--checkpoint` path.

## Benchmark export and paper table flow

Use the first-party paper benchmark spec:

```text
benchmarks/refexp_paper_benchmark.json
```

The benchmark runner scans `--pred-root` recursively for prediction JSON files. Keep that tree clean. It should contain only prediction payloads for the manifests declared by the spec you are scoring.

Verified JSON export:

```bash
conda run -n reasonseg-py311 python scripts/benchmark/run_benchmark.py --spec benchmarks/refexp_paper_benchmark.json --pred-root /home/lch/Project/ReasonSeg/.sisyphus/evidence/task-10-benchmark-preds --output /home/lch/Project/ReasonSeg/.sisyphus/evidence/task-10-refexp-paper-results.json
```

Verified CSV export:

```bash
conda run -n reasonseg-py311 python scripts/benchmark/run_benchmark.py --spec benchmarks/refexp_paper_benchmark.json --pred-root /home/lch/Project/ReasonSeg/.sisyphus/evidence/task-10-benchmark-preds --output /home/lch/Project/ReasonSeg/.sisyphus/evidence/task-10-refexp-paper-results.csv --output-format csv
```

How to use the outputs:

- JSON is the full nested report for archival and scripted post-processing.
  - top level suite summary: `suite_metrics`
  - per dataset summary: `dataset_metrics.refcoco_unc`, `dataset_metrics.refcoco_plus_unc`, `dataset_metrics.refcocog_umd`
  - per paper slice summary: `slice_metrics.noun`, `slice_metrics.attribute`, `slice_metrics.relation_action`, `slice_metrics.no_target`
- CSV is the flat table export for spreadsheets and paper tables.
  - `scope_type` tells you if a row is `suite`, `dataset`, or `slice`
  - `scope_name` is the dataset id or slice name
  - `metric_key` holds values such as `grounding/mIoU` or `no_target/rejection_rate`

The paper benchmark includes the three positive referring-expression families plus the `no_target` slice. A valid prediction export for this spec must cover all four manifest groups.

## Minimal handoff checklist

1. Verify data under `/home/lch/Project/ReasonSeg/datasets`.
2. Train into a fresh output root.
3. Run eval or test with an explicit checkpoint.
4. Collect only the prediction JSON files you want to score into a clean benchmark prediction root.
5. Export JSON for archive, CSV for paper tables.
