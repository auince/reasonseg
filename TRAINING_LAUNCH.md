# ReasonSeg Training Launch Guide

This document records the repo-local training startup commands and the key parameters used for the current RefCOCO runs.


cd /data/lch/Frames/ReasonSeg && python scripts/yolo_train.py \
  --task refcoco \
  --data-root /data/lch/Frames/ReasonSeg/dataset \
  --project /data/lch/Frames/ReasonSeg/outputs \
  --name refcoco_2gpu_run \
  --device 0,1 \
  --batch 2 \
  --lr 0.0002 \
  --opts TEST.EVAL_PERIOD 8497 SOLVER.CHECKPOINT_PERIOD 8497 



## Verified environment

- Repo root: `/home/lch/Project/ReasonSeg`
- Conda env: `reasonseg-py311`
- Dataset root: `/home/lch/Project/ReasonSeg/datasets`
- High-level launcher: `scripts/yolo_train.py`

All commands below are intended to be run from the repo root.

## Preferred high-level launcher

The repo now exposes a YOLOv8-style training entrypoint:

```bash
conda run -n reasonseg-py311 python scripts/yolo_train.py --help
```

This launcher wraps the internal stack:

- `scripts/yolo_train.py`
- `scripts/watch_train.py`
- `accelerate launch`
- `scripts/train.py`

So the user-facing command stays short while watchdog monitoring and multi-GPU launch stay enabled underneath.

## Core launcher arguments

- `--task`
  - dataset/config alias
  - supported values: `refcoco`, `refcoco+`, `refcocog`
- `--data-root`
  - local dataset root
- `--project`
  - parent output directory
- `--name`
  - run name under `--project`
- `--device`
  - single GPU like `0`, or multi-GPU like `0,1`
- `--batch`
  - global batch size passed to `SOLVER.IMS_PER_BATCH`
- `--lr`
  - learning rate override
- `--max-iter`
  - max training iterations override
- `--resume`
  - resume from `last_checkpoint`
- `--checkpoint`
  - explicit pretrained or resume checkpoint path
- `--opts`
  - raw config overrides in `KEY VALUE` form

## Latest valid dual-GPU command shape

This is the latest **valid** 2-GPU launcher shape in this repo. It is syntactically correct for Detectron2 distributed loading, but it proved unstable from VRAM pressure during training.

```bash
conda run -n reasonseg-py311 python scripts/yolo_train.py \
  --task refcoco \
  --data-root /home/lch/Project/ReasonSeg/datasets \
  --project /home/lch/Project/ReasonSeg/outputs \
  --name refcoco_2gpu_yolo_epoch_eval_run \
  --device 0,1 \
  --batch 2 \
  --lr 0.0002 \
  --max-iter 90000 \
  --opts TEST.EVAL_PERIOD 8497 SOLVER.CHECKPOINT_PERIOD 8497
```

## Invalid dual-GPU batch-1 attempt

This command is recorded as failure history only. Do not reuse it on 2 GPUs.

```bash
conda run -n reasonseg-py311 python scripts/yolo_train.py \
  --task refcoco \
  --data-root /home/lch/Project/ReasonSeg/datasets \
  --project /home/lch/Project/ReasonSeg/outputs \
  --name refcoco_2gpu_yolo_epoch_eval_b1_run \
  --device 0,1 \
  --batch 1 \
  --lr 0.0002 \
  --max-iter 90000 \
  --opts TEST.EVAL_PERIOD 16994 SOLVER.CHECKPOINT_PERIOD 16994
```

## Parameter notes for the current RefCOCO setup

- `--device 0,1`
  - launches 2-worker distributed training through `accelerate`
- `--batch 2`
  - sets `SOLVER.IMS_PER_BATCH=2`
- `TEST.EVAL_PERIOD 8497`
  - runs validation once per epoch for the current train set size when global batch is 2
- `SOLVER.CHECKPOINT_PERIOD 8497`
  - saves checkpoint once per epoch when global batch is 2
- `--max-iter 90000`
  - long-running training budget
- `--lr 0.0002`
  - current learning rate used in the recent dual-GPU runs

## Important multi-GPU batch constraint

For Detectron2-based distributed training in this repo, `--batch` maps to the **global** `SOLVER.IMS_PER_BATCH`, not per-GPU batch size.

That means:

- on `--device 0,1`, the batch value must be divisible by `2`
- `--batch 1` is **invalid** for 2 GPUs
- `--batch 2` means 1 image per GPU
- `--batch 4` means 2 images per GPU

The failed run `refcoco_2gpu_yolo_epoch_eval_b1_run` stopped at loader construction with:

```text
AssertionError: Total batch size (1) must be divisible by the number of gpus (2).
```

So when using 2 GPUs, always choose an even `--batch` value.

## Output layout

For a command like:

```bash
--project /home/lch/Project/ReasonSeg/outputs --name refcoco_2gpu_yolo_epoch_eval_run
```

the launcher writes to:

```text
/home/lch/Project/ReasonSeg/outputs/refcoco_2gpu_yolo_epoch_eval_run/
```

and training artifacts land under:

```text
/home/lch/Project/ReasonSeg/outputs/refcoco_2gpu_yolo_epoch_eval_run/run_0/
```

Important files:

- `run_0/config.yaml`
- `run_0/log.txt`
- `run_0/log.txt.rank1`
- `run_0/train_metrics.json`
- `run_0/model_final.pth`
- `run_0/last_checkpoint`
- `accelerate_logs/.../stdout.log`
- `accelerate_logs/.../stderr.log`

## Resume command shape

To resume a stopped run from its `last_checkpoint`:

```bash
conda run -n reasonseg-py311 python scripts/yolo_train.py \
  --task refcoco \
  --data-root /home/lch/Project/ReasonSeg/datasets \
  --project /home/lch/Project/ReasonSeg/outputs \
  --name refcoco_2gpu_yolo_epoch_eval_run \
  --device 0,1 \
  --batch 2 \
  --lr 0.0002 \
  --max-iter 90000 \
  --resume \
  --opts TEST.EVAL_PERIOD 8497 SOLVER.CHECKPOINT_PERIOD 8497
```

## Notes about monitoring and validation

- `scripts/watch_train.py` is always in the launch chain for this high-level entrypoint.
- The launcher keeps dual-GPU runs under watchdog supervision.
- Validation metrics are written into the train log when `TEST.EVAL_PERIOD` is nonzero.
- The current runtime also performs explicit memory cleanup after periodic eval before returning to training.

## Current run history summary

- `refcoco_2gpu_yolo_eval_run`
  - batch 2
  - eval every 500 iter
  - failed after periodic eval because training resumed into CUDA OOM
- `refcoco_2gpu_yolo_epoch_eval_run`
  - batch 2
  - eval once per epoch
  - still hit early training CUDA OOM before first eval
- `refcoco_2gpu_yolo_epoch_eval_b1_run`
  - batch 1 attempt on 2 GPUs
  - invalid because global batch 1 is not divisible by 2 GPUs
