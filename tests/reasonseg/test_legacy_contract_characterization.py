# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from reasonseg.data.dataset_mappers.refcoco_dataset_mapper import RefCOCODatasetMapper
from reasonseg.modeling._compat import CfgNode as CN
from reasonseg.runtime import common as runtime_common
from model.vr_ov_config import validate_vr_ov_config


_INCOMPLETE_VR_OV_CONFIG = Path(
    "/home/lch/Project/ReasonSeg/tests/fixtures/vr_ov_incomplete_missing_hidden_dim.yaml"
)


def test_reasonseg_mapper_characterizes_current_composed_prompt_behavior() -> None:
    mapper = RefCOCODatasetMapper(
        is_train=False,
        augmentations=[],
        image_format="RGB",
        metadata=object(),
        dataset_name="refcoco_val_unc",
        reasonseg_enabled=True,
    )
    dataset_dict: dict[str, object] = {}
    prompts = ["  The Red Dog  ", "man watering flowers", "no bicycle"]

    mapper._attach_reasonseg_fields(dataset_dict, prompts)

    assert dataset_dict["query_text"] == prompts
    assert dataset_dict["query_struct"] == [
        {
            "target": "dog",
            "attributes": ["red"],
            "relations": [],
            "actions": [],
            "negatives": [],
            "exists": True,
        },
        {
            "target": "man",
            "attributes": [],
            "relations": [],
            "actions": [{"verb": "watering", "target": "flowers"}],
            "negatives": [],
            "exists": True,
        },
        {
            "target": None,
            "attributes": [],
            "relations": [],
            "actions": [],
            "negatives": ["absent_object"],
            "exists": False,
        },
    ]
    assert dataset_dict["requested_target"] == ["dog", "man", "bicycle"]
    assert dataset_dict["slice_tags"] == ["attribute", "relation_action", "no_target"]
    assert dataset_dict["positive_mask_count"] == [1, 1, 0]
    assert dataset_dict["composed_prompt"] == [
        "red dog",
        "man watering flowers",
        "no bicycle",
    ]


def test_setup_cfg_characterizes_legacy_reasonseg_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reasonseg.data.runtime_refcoco as runtime_refcoco

    data_root = tmp_path / "datasets"
    data_root.mkdir()
    monkeypatch.setattr(runtime_refcoco, "register_refcoco_datasets", lambda data_root: None)

    cfg = runtime_common.setup_cfg(
        argparse.Namespace(
            config=Path("/home/lch/Project/ReasonSeg/configs/refcoco/refcoco_reasonseg.yaml"),
            data_root=data_root,
            output_dir=tmp_path / "outputs",
            run_index=0,
            checkpoint=None,
            batch_size=None,
            lr=None,
            max_iter=1,
            opts=[],
        )
    )

    assert cfg.MODEL.META_ARCHITECTURE == "OpenWorldSAM2"
    assert cfg.MODEL.OpenWorldSAM2.REASONSEG_ENABLED is True
    assert cfg.MODEL.OpenWorldSAM2.composition_mode == "composed_prompt"
    assert cfg.DATASETS.TRAIN == ("refcoco_train_unc",)
    assert cfg.DATASETS.TEST == ("refcoco_val_unc",)
    assert Path(cfg.OUTPUT_DIR) == (tmp_path / "outputs" / "run_0").resolve()


def test_setup_cfg_characterizes_exposed_vr_ov_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reasonseg.data.runtime_refcoco as runtime_refcoco

    data_root = tmp_path / "datasets"
    data_root.mkdir()
    monkeypatch.setattr(runtime_refcoco, "register_refcoco_datasets", lambda data_root: None)

    cfg = runtime_common.setup_cfg(
        argparse.Namespace(
            config=Path("/home/lch/Project/ReasonSeg/configs/vr_ov/vr_ov_base.yaml"),
            data_root=data_root,
            output_dir=tmp_path / "outputs",
            run_index=0,
            checkpoint=None,
            batch_size=None,
            lr=None,
            max_iter=1,
            opts=[],
        )
    )

    assert cfg.MODEL.META_ARCHITECTURE == "VR_OV"
    assert cfg.MODEL.VR_OV.ENABLED is True
    assert cfg.MODEL.VR_OV.QUERY_PARSER.ENABLED is True
    assert cfg.MODEL.VR_OV.QUERY_PARSER.HIDDEN_DIM == 256
    assert cfg.MODEL.VR_OV.QUERY_PARSER.OUT_DIM == 256
    assert cfg.MODEL.VR_OV.SCENE_GRAPH.ENABLED is True
    assert cfg.MODEL.VR_OV.SCENE_GRAPH.HIDDEN_DIM == 256
    assert cfg.MODEL.VR_OV.COMP_MATCHER.ENABLED is True
    assert cfg.MODEL.VR_OV.COMP_MATCHER.HIDDEN_DIM == 256
    assert cfg.MODEL.VR_OV.REFINE_DECODER.ENABLED is True
    assert cfg.MODEL.VR_OV.REFINE_DECODER.ATTR_THRESHOLD == 0.5
    assert Path(cfg.OUTPUT_DIR) == (tmp_path / "outputs" / "run_0").resolve()


def test_setup_cfg_rejects_vr_ov_partial_enablement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reasonseg.data.runtime_refcoco as runtime_refcoco

    data_root = tmp_path / "datasets"
    data_root.mkdir()
    monkeypatch.setattr(runtime_refcoco, "register_refcoco_datasets", lambda data_root: None)

    with pytest.raises(ValueError, match="partial enablement"):
        runtime_common.setup_cfg(
            argparse.Namespace(
                config=Path("/home/lch/Project/ReasonSeg/configs/vr_ov/vr_ov_base.yaml"),
                data_root=data_root,
                output_dir=tmp_path / "outputs",
                run_index=0,
                checkpoint=None,
                batch_size=None,
                lr=None,
                max_iter=1,
                opts=["MODEL.VR_OV.QUERY_PARSER.ENABLED", "False"],
            )
        )


def test_setup_cfg_rejects_vr_ov_meta_arch_without_enablement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reasonseg.data.runtime_refcoco as runtime_refcoco

    data_root = tmp_path / "datasets"
    data_root.mkdir()
    monkeypatch.setattr(runtime_refcoco, "register_refcoco_datasets", lambda data_root: None)

    with pytest.raises(ValueError, match='META_ARCHITECTURE="VR_OV"'):
        runtime_common.setup_cfg(
            argparse.Namespace(
                config=Path("/home/lch/Project/ReasonSeg/configs/vr_ov/vr_ov_base.yaml"),
                data_root=data_root,
                output_dir=tmp_path / "outputs",
                run_index=0,
                checkpoint=None,
                batch_size=None,
                lr=None,
                max_iter=1,
                opts=["MODEL.VR_OV.ENABLED", "False"],
            )
        )


def test_validate_vr_ov_config_rejects_missing_canonical_fields() -> None:
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.META_ARCHITECTURE = "VR_OV"
    cfg.MODEL.VR_OV = CN()
    cfg.MODEL.VR_OV.ENABLED = True
    cfg.MODEL.VR_OV.QUERY_PARSER = CN()
    cfg.MODEL.VR_OV.QUERY_PARSER.ENABLED = True
    cfg.MODEL.VR_OV.QUERY_PARSER.GNN_LAYERS = 2
    cfg.MODEL.VR_OV.QUERY_PARSER.GNN_HEADS = 4
    cfg.MODEL.VR_OV.QUERY_PARSER.CHECKPOINT = ""
    cfg.MODEL.VR_OV.QUERY_PARSER.OUT_DIM = 256
    cfg.MODEL.VR_OV.SCENE_GRAPH = CN()
    cfg.MODEL.VR_OV.SCENE_GRAPH.ENABLED = True
    cfg.MODEL.VR_OV.SCENE_GRAPH.HOI_TOKENS = 5
    cfg.MODEL.VR_OV.SCENE_GRAPH.REGION_TOPK = 64
    cfg.MODEL.VR_OV.SCENE_GRAPH.HIDDEN_DIM = 256
    cfg.MODEL.VR_OV.COMP_MATCHER = CN()
    cfg.MODEL.VR_OV.COMP_MATCHER.ENABLED = True
    cfg.MODEL.VR_OV.COMP_MATCHER.CMF_LAYERS = 3
    cfg.MODEL.VR_OV.COMP_MATCHER.HIDDEN_DIM = 256
    cfg.MODEL.VR_OV.REFINE_DECODER = CN()
    cfg.MODEL.VR_OV.REFINE_DECODER.ENABLED = True
    cfg.MODEL.VR_OV.REFINE_DECODER.MAX_ITER = 3
    cfg.MODEL.VR_OV.REFINE_DECODER.ATTR_THRESHOLD = 0.5

    with pytest.raises(ValueError, match="QUERY_PARSER.HIDDEN_DIM"):
        validate_vr_ov_config(cfg)


def test_setup_cfg_rejects_incomplete_canonical_vr_ov_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reasonseg.data.runtime_refcoco as runtime_refcoco

    data_root = tmp_path / "datasets"
    data_root.mkdir()
    monkeypatch.setattr(runtime_refcoco, "register_refcoco_datasets", lambda data_root: None)

    with pytest.raises(ValueError, match="QUERY_PARSER.HIDDEN_DIM"):
        runtime_common.setup_cfg(
            argparse.Namespace(
                config=_INCOMPLETE_VR_OV_CONFIG,
                data_root=data_root,
                output_dir=tmp_path / "outputs",
                run_index=0,
                checkpoint=None,
                batch_size=None,
                lr=None,
                max_iter=1,
                opts=[],
            )
        )


def test_runtime_helpers_characterize_legacy_train_and_eval_artifact_layout(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(output_dir=tmp_path / "outputs", run_index=0)

    output_dir = runtime_common.get_output_dir(args)
    inference_dir = runtime_common.get_inference_output_dir(output_dir)
    checkpoint_path = output_dir / "model_final.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("checkpoint\n", encoding="utf-8")

    checkpoint_record = runtime_common.write_last_checkpoint(output_dir, checkpoint_path)

    assert output_dir == (tmp_path / "outputs" / "run_0").resolve()
    assert runtime_common.get_config_dump_path(output_dir) == output_dir / "config.yaml"
    assert runtime_common.get_train_metrics_path(output_dir) == output_dir / "train_metrics.json"
    assert checkpoint_record == output_dir / "last_checkpoint"
    assert checkpoint_record.read_text(encoding="utf-8") == "model_final.pth\n"
    assert inference_dir == output_dir / "inference"
    assert inference_dir / runtime_common.PREDICTIONS_NAME == output_dir / "inference" / "predictions.json"
    assert inference_dir / runtime_common.METRICS_NAME == output_dir / "inference" / "metrics.json"
