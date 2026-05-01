from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from model.vr_ov import VR_OV
from model.vr_ov_types import CompositionScores
from model.vr_ov_types import QueryGraph
from reasonseg.modeling import open_world_sam2 as open_world_sam2_module
from reasonseg.runtime import common as runtime_common
from reasonseg.runtime import train as runtime_train


class _TinyVRModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vr_ov_scene_graph = torch.nn.Linear(2, 2)
        self.vr_ov_comp_matcher = torch.nn.Linear(2, 2)
        self.vr_ov_refine_decoder = torch.nn.Linear(2, 2)
        self.backbone = torch.nn.Linear(2, 2)


def _make_query_graph() -> QueryGraph:
    return QueryGraph(
        nodes=[torch.ones(4) * (index + 1) for index in range(4)],
        edges=torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]], dtype=torch.long),
        node_types=["category", "attribute", "relation", "action"],
    )


def test_apply_vr_ov_phase_freezing_keeps_only_phase_module_trainable() -> None:
    model = _TinyVRModel()

    summary = runtime_train.apply_vr_ov_phase_freezing(model, "1b")

    trainable_names = set(summary["trainable_names"])
    assert trainable_names == {
        "vr_ov_comp_matcher.weight",
        "vr_ov_comp_matcher.bias",
    }
    assert model.vr_ov_comp_matcher.weight.requires_grad
    assert not model.vr_ov_scene_graph.weight.requires_grad
    assert not model.vr_ov_refine_decoder.weight.requires_grad
    assert not model.backbone.weight.requires_grad


def test_resolve_vr_ov_curriculum_state_defaults_to_even_progression() -> None:
    state = runtime_train.resolve_vr_ov_curriculum_state(
        argparse.Namespace(curriculum_levels=["L1", "L2", "L3", "L4"], curriculum_switch_interval=0),
        max_iter=10,
    )

    assert state.switch_interval == 2
    assert [state.level_for_iteration(index) for index in (0, 2, 4, 8)] == [
        "L1",
        "L2",
        "L3",
        "L4",
    ]


def test_filter_batch_by_vr_ov_curriculum_uses_query_complexity_levels() -> None:
    batch = [
        {
            "prompt": ["dog", "red dog", "dog left of car", "no bicycle"],
            "instances": [1, 2, 3, 4],
            "query_struct": [
                {"exists": True, "attributes": [], "relations": [], "actions": [], "negatives": []},
                {"exists": True, "attributes": ["red"], "relations": [], "actions": [], "negatives": []},
                {"exists": True, "attributes": [], "relations": [{"type": "left_of", "object": "car"}], "actions": [], "negatives": []},
                {"exists": False, "attributes": [], "relations": [], "actions": [], "negatives": ["absent_object"]},
            ],
            "slice_tags": ["noun", "attribute", "relation_action", "no_target"],
            "query_metadata": [
                {"image_id": 10, "prompt_index": 0, "prompt_count": 4},
                {"image_id": 10, "prompt_index": 1, "prompt_count": 4},
                {"image_id": 10, "prompt_index": 2, "prompt_count": 4},
                {"image_id": 10, "prompt_index": 3, "prompt_count": 4},
            ],
            "composed_prompt": ["dog", "red dog", "dog left of car", "no bicycle"],
        }
    ]

    filtered_batch, batch_metrics = runtime_train.filter_batch_by_vr_ov_curriculum(
        batch,
        current_level="L2",
    )

    assert filtered_batch[0]["prompt"] == ["dog", "red dog"]
    assert filtered_batch[0]["instances"] == [1, 2]
    assert filtered_batch[0]["query_metadata"] == [
        {"image_id": 10, "prompt_index": 0, "prompt_count": 4},
        {"image_id": 10, "prompt_index": 1, "prompt_count": 4},
    ]
    assert batch_metrics.total_prompts == 4
    assert batch_metrics.kept_prompts == 2
    assert batch_metrics.forced_prompt_keeps == 0


def test_filter_batch_by_vr_ov_curriculum_keeps_query_metadata_per_sample() -> None:
    batch = [
        {
            "prompt": ["dog", "red dog"],
            "query_text": ["dog", "red dog"],
            "requested_target": ["dog", "dog"],
            "positive_mask_count": [1, 1],
            "instances": [11, 12],
            "query_struct": [
                {"exists": True, "attributes": [], "relations": [], "actions": [], "negatives": []},
                {"exists": True, "attributes": ["red"], "relations": [], "actions": [], "negatives": []},
            ],
            "slice_tags": ["noun", "attribute"],
            "query_metadata": [
                {"image_id": 101, "prompt_index": 0, "prompt_count": 2},
                {"image_id": 101, "prompt_index": 1, "prompt_count": 2},
            ],
        },
        {
            "prompt": ["cat behind chair", "blue cat"],
            "query_text": ["cat behind chair", "blue cat"],
            "requested_target": ["cat", "cat"],
            "positive_mask_count": [1, 1],
            "instances": [21, 22],
            "query_struct": [
                {"exists": True, "attributes": [], "relations": [{"type": "behind", "target": "chair"}], "actions": [], "negatives": []},
                {"exists": True, "attributes": ["blue"], "relations": [], "actions": [], "negatives": []},
            ],
            "slice_tags": ["relation_action", "attribute"],
            "query_metadata": [
                {"image_id": 202, "prompt_index": 0, "prompt_count": 2},
                {"image_id": 202, "prompt_index": 1, "prompt_count": 2},
            ],
        },
    ]

    filtered_batch, batch_metrics = runtime_train.filter_batch_by_vr_ov_curriculum(
        batch,
        current_level="L2",
    )

    assert filtered_batch[0]["prompt"] == ["dog", "red dog"]
    assert filtered_batch[0]["query_metadata"] == [
        {"image_id": 101, "prompt_index": 0, "prompt_count": 2},
        {"image_id": 101, "prompt_index": 1, "prompt_count": 2},
    ]
    assert filtered_batch[1]["prompt"] == ["blue cat"]
    assert filtered_batch[1]["query_text"] == ["blue cat"]
    assert filtered_batch[1]["instances"] == [22]
    assert filtered_batch[1]["query_metadata"] == [
        {"image_id": 202, "prompt_index": 1, "prompt_count": 2},
    ]
    assert batch_metrics.total_prompts == 4
    assert batch_metrics.kept_prompts == 3


def test_clone_query_graph_with_dropout_zeroes_dropped_nodes_but_keeps_shape() -> None:
    query_graph = _make_query_graph()
    generator = torch.Generator().manual_seed(0)

    dropped_graph, kept_nodes, total_nodes = runtime_train._clone_query_graph_with_dropout(
        query_graph,
        dropout_p=0.2,
        generator=generator,
    )

    assert total_nodes == 4
    assert kept_nodes == 2
    assert len(dropped_graph.nodes) == 4
    zeroed = [torch.count_nonzero(node).item() == 0 for node in dropped_graph.nodes]
    assert sum(zeroed) == 2


def test_install_vr_ov_query_dropout_records_observable_metrics() -> None:
    class _Parser(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return _make_query_graph()

    model = SimpleNamespace(vr_ov_query_parser=_Parser())
    controller = runtime_train.VROVRuntimeController(
        argparse.Namespace(
            phase="1a",
            curriculum_levels=["L1", "L2", "L3", "L4"],
            curriculum_switch_interval=1,
            query_dropout_p=0.2,
            ema_decay=0.999,
            vr_ov_seed=0,
        ),
        max_iter=8,
    )

    runtime_train.install_vr_ov_query_dropout(model, controller)
    dropped = model.vr_ov_query_parser.forward()

    assert isinstance(dropped, QueryGraph)
    assert controller.metrics.dropout_events == 1
    assert controller.metrics.query_nodes_seen == 4
    assert controller.metrics.query_nodes_kept == 2


def test_model_ema_state_updates_and_round_trips() -> None:
    model = torch.nn.Linear(2, 2)
    initial_weight = model.weight.detach().clone()
    ema_state = runtime_train.ModelEmaState(model, decay=0.5)

    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(4.0)
    ema_state.update(model)
    saved = ema_state.state_dict()

    restored = runtime_train.ModelEmaState(torch.nn.Linear(2, 2), decay=0.1)
    restored.load_state_dict(saved)

    assert restored.decay == 0.5
    assert restored.num_updates == 1
    expected_weight = initial_weight * 0.5 + torch.full_like(initial_weight, 2.0) * 0.5
    assert torch.allclose(restored.shadow_params["weight"], expected_weight)


def test_vr_ov_train_worker_writes_extended_metrics_payload(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "outputs" / "run_0"
    cfg = SimpleNamespace(
        OUTPUT_DIR=str(output_dir),
        SOLVER=SimpleNamespace(MAX_ITER=2, CHECKPOINT_PERIOD=10, IMS_PER_BATCH=2, BASE_LR=0.001),
        DATASETS=SimpleNamespace(TRAIN=("refcoco_train_unc",), TEST=()),
        MODEL=SimpleNamespace(WEIGHTS="cfg-model-weights.pth"),
        TEST=SimpleNamespace(EVAL_PERIOD=0),
        SEED=-1,
        CUDNN_BENCHMARK=False,
        dump=lambda: "OUTPUT_DIR: test\n",
    )

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vr_ov_scene_graph = torch.nn.Linear(1, 1)
            self.vr_ov_comp_matcher = torch.nn.Linear(1, 1)
            self.vr_ov_refine_decoder = torch.nn.Linear(1, 1)
            self.backbone = torch.nn.Linear(1, 1)

        def forward(self, batch):
            value = self.vr_ov_scene_graph.weight.sum() * 0
            value = value + torch.tensor(float(len(batch[0]["prompt"])), dtype=torch.float32)
            return {"loss_total": value}

    class _Scheduler:
        def step(self) -> None:
            return None

    class _Checkpointer:
        def __init__(self, model, *, save_dir: str, optimizer, scheduler, ema_state=None) -> None:
            self.save_dir = save_dir

        def resume_or_load(self, path, *, resume: bool):
            return {}

        def save(self, name: str, **kwargs) -> None:
            save_path = Path(self.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / f"{name}.pth").write_text("checkpoint\n", encoding="utf-8")

    monkeypatch.setattr(runtime_common, "setup_cfg", lambda args: cfg)
    monkeypatch.setattr(runtime_common, "setup_runtime_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_common, "maybe_wrap_model", lambda wrapped_model: wrapped_model)
    monkeypatch.setattr(
        runtime_common,
        "build_refcoco_train_loader",
        lambda cfg: [
            [
                {
                    "prompt": ["dog", "red dog", "dog left of car", "no bicycle"],
                    "instances": [1, 2, 3, 4],
                    "query_struct": [
                        {"exists": True, "attributes": [], "relations": [], "actions": [], "negatives": []},
                        {"exists": True, "attributes": ["red"], "relations": [], "actions": [], "negatives": []},
                        {"exists": True, "attributes": [], "relations": [{"type": "left_of", "object": "car"}], "actions": [], "negatives": []},
                        {"exists": False, "attributes": [], "relations": [], "actions": [], "negatives": ["absent_object"]},
                    ],
                    "slice_tags": ["noun", "attribute", "relation_action", "no_target"],
                    "composed_prompt": ["dog", "red dog", "dog left of car", "no bicycle"],
                }
            ]
        ],
    )
    monkeypatch.setattr(
        runtime_common,
        "build_optimizer",
        lambda cfg, model: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(
        runtime_common,
        "_import_runtime_deps",
        lambda: {
            "DetectionCheckpointer": _Checkpointer,
            "build_lr_scheduler": lambda cfg, optimizer: _Scheduler(),
            "build_model": lambda cfg: _Model(),
            "comm": SimpleNamespace(reduce_dict=lambda loss_dict: loss_dict, is_main_process=lambda: True),
        },
    )

    runtime_train._worker(
        argparse.Namespace(
            resume=False,
            checkpoint=None,
            vr_ov_enabled=True,
            phase="1a",
            curriculum_levels=["L1", "L2", "L3", "L4"],
            curriculum_switch_interval=1,
            query_dropout_p=0.2,
            ema_decay=0.999,
            vr_ov_seed=0,
        )
    )

    payload = __import__("json").loads((output_dir / "train_metrics.json").read_text())
    assert payload["loss_history"] == [1.0, 2.0]
    assert payload["vr_ov"]["phase"] == "1a"
    assert payload["vr_ov"]["freeze_summary"]["trainable_param_count"] == 2
    assert payload["vr_ov"]["ema_updates"] == 2


def test_setup_cfg_for_train_vr_ov_enables_phase_modules_from_vr_ov_base(
    tmp_path: Path, monkeypatch
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
    assert cfg.MODEL.VR_OV.COMP_MATCHER.ENABLED is True
    assert cfg.MODEL.VR_OV.REFINE_DECODER.ENABLED is True


def test_vr_ov_from_config_passes_live_hidden_size_to_vr_ov_parser(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    class _FakeTokenizer:
        eos_token_id = 2
        bos_token_id = 0
        pad_token_id = 1

    class _FakeEvfModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(hidden_size=1024, eos_token_id=None, bos_token_id=None, pad_token_id=None)
            self.visual_model = SimpleNamespace(memory_attention=SimpleNamespace(d_model=256))
            self.mm_extractor = torch.nn.Linear(1, 1)
            self.text_hidden_fcs = torch.nn.ModuleList([torch.nn.Linear(1, 1)])

    class _FakeParser(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            seen["parser_kwargs"] = kwargs

    monkeypatch.setattr(open_world_sam2_module, "_load_tokenizer", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr(
        open_world_sam2_module.EvfSam2Model,
        "from_pretrained",
        lambda *args, **kwargs: _FakeEvfModel(),
    )
    monkeypatch.setattr(open_world_sam2_module, "HungarianMatcher", lambda **kwargs: object())
    monkeypatch.setattr(open_world_sam2_module, "SetCriterion", lambda **kwargs: object())
    monkeypatch.setattr(open_world_sam2_module.MetadataCatalog, "get", lambda name: object())
    query_parser_module = __import__(
        "model.query_parser", fromlist=["BIOQueryParser", "LLMQueryParser"]
    )
    monkeypatch.setattr(query_parser_module, "BIOQueryParser", _FakeParser)
    monkeypatch.setattr(query_parser_module, "LLMQueryParser", _FakeParser)

    cfg = SimpleNamespace(
        MODEL=SimpleNamespace(
            PIXEL_MEAN=[0.0, 0.0, 0.0],
            PIXEL_STD=[1.0, 1.0, 1.0],
            OpenWorldSAM2=SimpleNamespace(
                TORCH_DTYPE="fp32",
                TRAIN_MASK_DECODER=False,
                TRAIN_PROMPT_ENCODER=False,
                VISION_PRETRAINED="checkpoints/sam_vit_h_4b8939.pth",
                ENCODER_PRETRAINED="checkpoints/beit3_large_patch16_224.pth",
                HF_LOCAL_FILES_ONLY=True,
                TOKENIZER_CONFIG="unused",
                LOCAL_TOKENIZER_CONFIG="unused",
                EVF_CONFIG="unused",
                LOCAL_EVF_CONFIG="unused",
                TRAIN_VLM=True,
                QUERY_DIM=256,
                NUM_OBJECT_QUERIES=20,
                TRAIN_TIE_BREAKER=True,
                NO_OBJECT_WEIGHT=0.1,
                DICE_WEIGHT=1.0,
                MASK_WEIGHT=5.0,
                OBJECTNESS_WEIGHT=1.0,
                USE_CROSS_ATTENTION=False,
                USE_VISUAL_TOKENS=True,
                SAM_IOU=True,
                REASONSEG_ENABLED=False,
                composition_mode="composed_prompt",
                CROSS_ATTENTION_LAYERS=1,
                LEARNED_PARSER=SimpleNamespace(ENABLED=False),
                TEST=SimpleNamespace(
                    TWO_STAGE_INFERENCE=False,
                    REFER_ON=False,
                    SEMANTIC_ON=False,
                    INSTANCE_ON=True,
                    PANOPTIC_ON=False,
                    TOP_K_ON=False,
                    NMS_ON=False,
                    DETECTIONS_PER_IMAGE=20,
                    NMS_THRESHOLD=0.0,
                    IOU_THRESHOLD=0.0,
                ),
            ),
            VR_OV=SimpleNamespace(
                ENABLED=True,
                QUERY_PARSER=SimpleNamespace(
                    ENABLED=True,
                    CHECKPOINT="",
                    GNN_LAYERS=2,
                    GNN_HEADS=4,
                    HIDDEN_DIM=256,
                    OUT_DIM=256,
                ),
                SCENE_GRAPH=SimpleNamespace(ENABLED=False, HIDDEN_DIM=256, HOI_TOKENS=5, REGION_TOPK=64),
                COMP_MATCHER=SimpleNamespace(ENABLED=False, HIDDEN_DIM=256, CMF_LAYERS=3),
                REFINE_DECODER=SimpleNamespace(ENABLED=False, MAX_ITER=3, ATTR_THRESHOLD=0.5),
            ),
        ),
        DATASETS=SimpleNamespace(TRAIN=("refcoco_train_unc",)),
    )

    result = VR_OV.from_config(cfg)
    model = VR_OV(**result)

    assert isinstance(model, VR_OV)
    assert isinstance(result["vr_ov_query_parser"], _FakeParser)
    assert isinstance(model.vr_ov_query_parser, _FakeParser)
    assert seen["parser_kwargs"] == {
        "parser_checkpoint": None,
        "hidden_dim": 1024,
        "num_layers": 2,
        "nhead": 4,
        "gnn_hidden": 256,
        "gnn_out": 256,
    }


def test_open_world_sam2_from_config_keeps_legacy_build_path_free_of_vr_ov_modules(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        open_world_sam2_module,
        "_build_open_world_sam2_common_kwargs",
        lambda cfg: ({"tokenizer": "base"}, 1024),
    )
    monkeypatch.setattr(
        open_world_sam2_module,
        "_build_open_world_sam2_parser_kwargs",
        lambda cfg: {"parser_head": "parser"},
    )

    result = open_world_sam2_module.OpenWorldSAM2.from_config(SimpleNamespace())

    assert result == {"tokenizer": "base", "parser_head": "parser"}
    assert "vr_ov_query_parser" not in result


def test_scene_graph_prompt_context_creates_gradient_path() -> None:
    batch_feat_with_tokens = torch.randn(20, 1, 256, requires_grad=True)
    hoi_tokens = torch.randn(5, 256, requires_grad=True)

    augmented = open_world_sam2_module.OpenWorldSAM2._apply_vr_ov_scene_graph_prompt_context(
        batch_feat_with_tokens,
        hoi_tokens,
    )
    loss = augmented.sum()
    loss.backward()

    assert hoi_tokens.grad is not None
    assert torch.count_nonzero(hoi_tokens.grad).item() > 0


def test_run_vr_ov_comp_matcher_batches_query_graphs_per_prompt_without_reuse() -> None:
    seen: dict[str, object] = {}

    class _Matcher:
        def __call__(self, query_nodes, visual_features, img_feat):
            seen["query_nodes_shape"] = tuple(query_nodes.shape)
            seen["visual_features_shape"] = tuple(visual_features.shape)
            seen["img_feat_shape"] = tuple(img_feat.shape)
            seen["category_markers"] = [
                float(query_nodes[index, 0, 0].item())
                for index in range(query_nodes.shape[0])
            ]
            batch_size = query_nodes.shape[0]
            _, _, height, width = img_feat.shape
            return (
                CompositionScores(
                    cat_feat=torch.ones(batch_size, 1, height, width),
                    attr_feat=torch.ones(batch_size, 1, height, width),
                    rel_feat=torch.ones(batch_size, 1, height, width),
                    act_feat=torch.ones(batch_size, 1, height, width),
                ),
                img_feat,
            )

    model = SimpleNamespace(
        vr_ov_comp_matcher=_Matcher(),
        _new_composition_scores=lambda: CompositionScores(),
        _validate_query_graph=open_world_sam2_module.OpenWorldSAM2._validate_query_graph,
        _validate_composition_scores=open_world_sam2_module.OpenWorldSAM2._validate_composition_scores,
    )
    query_graphs = [
        QueryGraph(
            nodes=[torch.tensor([1.0, 0.0]), torch.zeros(2), torch.zeros(2), torch.zeros(2)],
            edges=torch.tensor([[0], [1]], dtype=torch.long),
            node_types=["category", "attribute", "relation", "action"],
        ),
        QueryGraph(
            nodes=[torch.tensor([2.0, 0.0]), torch.zeros(2), torch.zeros(2), torch.zeros(2)],
            edges=torch.tensor([[0], [1]], dtype=torch.long),
            node_types=["category", "attribute", "relation", "action"],
        ),
    ]
    image_embed = torch.arange(8, dtype=torch.float32).view(2, 2, 2)

    comp_scores = open_world_sam2_module.OpenWorldSAM2._run_vr_ov_comp_matcher(
        model,
        query_graphs=query_graphs,
        image_embed=image_embed,
    )

    assert isinstance(comp_scores, CompositionScores)
    assert seen["query_nodes_shape"] == (2, 4, 2)
    assert seen["visual_features_shape"] == (2, 4, 2)
    assert seen["img_feat_shape"] == (2, 2, 2, 2)
    assert seen["category_markers"] == [1.0, 2.0]


def test_run_vr_ov_comp_matcher_rejects_partial_outputs() -> None:
    class _PartialMatcher:
        def __call__(self, query_nodes, visual_features, img_feat):
            batch_size = query_nodes.shape[0]
            _, _, height, width = img_feat.shape
            return (
                CompositionScores(
                    cat_feat=torch.ones(batch_size, 1, height, width),
                    attr_feat=None,
                    rel_feat=torch.ones(batch_size, 1, height, width),
                    act_feat=torch.ones(batch_size, 1, height, width),
                ),
                img_feat,
            )

    model = SimpleNamespace(
        vr_ov_comp_matcher=_PartialMatcher(),
        _new_composition_scores=lambda: CompositionScores(),
        _validate_query_graph=open_world_sam2_module.OpenWorldSAM2._validate_query_graph,
        _validate_composition_scores=open_world_sam2_module.OpenWorldSAM2._validate_composition_scores,
    )
    query_graphs = [_make_query_graph()]
    image_embed = torch.arange(8, dtype=torch.float32).view(2, 2, 2)

    with pytest.raises(ValueError, match="populate 'attr_feat'"):
        open_world_sam2_module.OpenWorldSAM2._run_vr_ov_comp_matcher(
            model,
            query_graphs=query_graphs,
            image_embed=image_embed,
        )


def test_build_processed_result_expands_prompt_level_scores_per_mask() -> None:
    seen: dict[str, list[float]] = {}

    class _RefineDecoder:
        def __call__(self, coarse_mask, comp_scores, visual_feat):
            seen["attr_markers"] = [
                float(comp_scores.attr_feat[index, 0, 0, 0].item())
                for index in range(comp_scores.attr_feat.shape[0])
            ]
            return coarse_mask + comp_scores.attr_feat, []

    model = SimpleNamespace(
        vr_ov_refine_decoder=_RefineDecoder(),
        num_tokens=2,
        refer_on=True,
        instance_on=False,
        panoptic_on=False,
        semantic_on=False,
        metadata=SimpleNamespace(stuff_classes=[]),
        _resize_composition_scores=open_world_sam2_module.OpenWorldSAM2._resize_composition_scores,
        _validate_composition_scores=open_world_sam2_module.OpenWorldSAM2._validate_composition_scores,
        _expand_composition_scores_for_masks=open_world_sam2_module.OpenWorldSAM2._expand_composition_scores_for_masks,
        postprocess_masks=lambda masks, orig_hw: masks,
        refer_inference=lambda pred_masks, pred_logits, class_labels: (pred_masks, pred_logits.squeeze(1)),
    )
    low_res_masks = torch.zeros(4, 1, 2, 2)
    pred_logits = torch.arange(4, dtype=torch.float32).view(4, 1)
    class_labels = torch.zeros(4, dtype=torch.int64)
    features = {
        "high_res_feats": [torch.ones(1, 32, 2, 2)],
        "image_embed": torch.ones(1, 256, 2, 2),
    }
    comp_scores = CompositionScores(
        cat_feat=torch.zeros(2, 1, 2, 2),
        attr_feat=torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[2.0, 2.0], [2.0, 2.0]]],
            ]
        ),
        rel_feat=torch.zeros(2, 1, 2, 2),
        act_feat=torch.zeros(2, 1, 2, 2),
    )

    result = open_world_sam2_module.OpenWorldSAM2._build_processed_result(
        model,
        low_res_masks=low_res_masks,
        pred_logits=pred_logits,
        class_labels=class_labels,
        features=features,
        high_res_features=[torch.ones(1, 256, 2, 2)],
        original_hw=(2, 2),
        img_idx=0,
        comp_scores=comp_scores,
        use_refine_decoder=True,
    )

    assert seen["attr_markers"] == [1.0, 1.0, 2.0, 2.0]
    assert torch.equal(
        result["grounding_mask"],
        torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[2.0, 2.0], [2.0, 2.0]]],
                [[[2.0, 2.0], [2.0, 2.0]]],
            ]
        ),
    )
    assert torch.equal(result["grounding_scores"], pred_logits.squeeze(1))


def test_build_processed_result_resizes_composition_scores_to_mask_resolution() -> None:
    seen: dict[str, tuple[int, ...]] = {}

    class _RefineDecoder:
        def __call__(self, coarse_mask, comp_scores, visual_feat):
            seen["attr_shape"] = tuple(comp_scores.attr_feat.shape)
            return coarse_mask + comp_scores.attr_feat, []

    model = SimpleNamespace(
        vr_ov_refine_decoder=_RefineDecoder(),
        num_tokens=1,
        refer_on=True,
        instance_on=False,
        panoptic_on=False,
        semantic_on=False,
        metadata=SimpleNamespace(stuff_classes=[]),
        _resize_composition_scores=open_world_sam2_module.OpenWorldSAM2._resize_composition_scores,
        _validate_composition_scores=open_world_sam2_module.OpenWorldSAM2._validate_composition_scores,
        _expand_composition_scores_for_masks=open_world_sam2_module.OpenWorldSAM2._expand_composition_scores_for_masks,
        postprocess_masks=lambda masks, orig_hw: masks,
        refer_inference=lambda pred_masks, pred_logits, class_labels: (pred_masks, pred_logits.squeeze(1)),
    )
    low_res_masks = torch.zeros(1, 1, 4, 4)
    pred_logits = torch.ones(1, 1)
    class_labels = torch.zeros(1, dtype=torch.int64)
    features = {
        "high_res_feats": [torch.ones(1, 32, 4, 4)],
        "image_embed": torch.ones(1, 256, 2, 2),
    }
    comp_scores = CompositionScores(
        cat_feat=torch.ones(1, 1, 2, 2),
        attr_feat=torch.ones(1, 1, 2, 2),
        rel_feat=torch.ones(1, 1, 2, 2),
        act_feat=torch.ones(1, 1, 2, 2),
    )

    result = open_world_sam2_module.OpenWorldSAM2._build_processed_result(
        model,
        low_res_masks=low_res_masks,
        pred_logits=pred_logits,
        class_labels=class_labels,
        features=features,
        high_res_features=[torch.ones(1, 256, 4, 4)],
        original_hw=(4, 4),
        img_idx=0,
        comp_scores=comp_scores,
        use_refine_decoder=True,
    )

    assert seen["attr_shape"] == (1, 1, 4, 4)
    assert torch.equal(result["grounding_mask"], torch.ones(1, 1, 4, 4))


def test_phase1a_freeze_configuration_still_allows_backward_into_scene_graph() -> None:
    class _Phase1aModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vr_ov_scene_graph = torch.nn.Linear(256, 256)
            self.backbone = torch.nn.Linear(256, 256)

        def forward(self) -> torch.Tensor:
            batch_feat = torch.randn(20, 1, 256)
            hoi_seed = self.vr_ov_scene_graph(torch.randn(5, 256))
            augmented = open_world_sam2_module.OpenWorldSAM2._apply_vr_ov_scene_graph_prompt_context(
                batch_feat,
                hoi_seed,
            )
            return augmented.sum()

    model = _Phase1aModel()
    runtime_train.apply_vr_ov_phase_freezing(model, "1a")

    loss = model.forward()
    loss.backward()

    assert model.vr_ov_scene_graph.weight.grad is not None
    assert model.backbone.weight.grad is None
