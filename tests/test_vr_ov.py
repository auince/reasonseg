# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import model.vr_ov as vr_ov_module
from model.vr_ov import VR_OV
from model.vr_ov_types import CompositionScores, QueryGraph
from reasonseg.modeling._compat import META_ARCH_REGISTRY
from reasonseg.modeling import open_world_sam2 as open_world_sam2_module
from reasonseg.modeling.open_world_sam2 import OpenWorldSAM2


class TestVR_OV:
    def test_registry(self) -> None:
        assert META_ARCH_REGISTRY.get("VR_OV") is VR_OV, (
            f"VR_OV not found in META_ARCH_REGISTRY. "
            f"Registered names: {[k for k in META_ARCH_REGISTRY]}"
        )

    def test_inheritance(self) -> None:
        assert issubclass(VR_OV, OpenWorldSAM2)
        assert issubclass(VR_OV, nn.Module)

    def test_import_clean(self) -> None:
        from model.vr_ov import VR_OV as imported_vr_ov

        assert imported_vr_ov is VR_OV
        assert imported_vr_ov.__name__ == "VR_OV"

    def test_vr_ov_owns_forward_and_from_config(self) -> None:
        assert VR_OV.forward is not OpenWorldSAM2.forward
        assert VR_OV.from_config.__func__ is not OpenWorldSAM2.from_config.__func__

    def test_from_config_merges_shared_and_vr_ov_build_kwargs(self, monkeypatch) -> None:
        calls: list[tuple[str, object]] = []

        monkeypatch.setattr(
            vr_ov_module,
            "_build_open_world_sam2_common_kwargs",
            lambda cfg: (calls.append(("common", cfg)) or ({"tokenizer": "base"}, 1024)),
        )
        monkeypatch.setattr(
            vr_ov_module,
            "_build_open_world_sam2_parser_kwargs",
            lambda cfg: calls.append(("parser", cfg)) or {"parser_head": "parser"},
        )
        monkeypatch.setattr(
            vr_ov_module,
            "_build_vr_ov_module_kwargs",
            lambda cfg, *, evf_hidden_size: calls.append(("vr_ov", evf_hidden_size))
            or {"vr_ov_query_parser": "vr-only"},
        )

        cfg = SimpleNamespace()
        result = VR_OV.from_config(cfg)

        assert result == {
            "tokenizer": "base",
            "parser_head": "parser",
            "vr_ov_query_parser": "vr-only",
        }
        assert calls == [
            ("common", cfg),
            ("parser", cfg),
            ("vr_ov", 1024),
        ]

    def test_open_world_sam2_from_config_does_not_reach_vr_ov_builder(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            vr_ov_module,
            "_build_vr_ov_module_kwargs",
            lambda *args, **kwargs: pytest.fail("VR_OV builder should not run here"),
        )

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

        result = OpenWorldSAM2.from_config(SimpleNamespace())

        assert result == {"tokenizer": "base", "parser_head": "parser"}

    def test_canonical_vr_ov_query_parser_uses_query_struct_per_prompt_without_composed_prompt(
        self,
    ) -> None:
        class _Tokenizer:
            @staticmethod
            def convert_ids_to_tokens(token_ids):
                return [f"tok-{token_id}" for token_id in token_ids]

        class _Parser:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __call__(self, **kwargs):
                query_struct = kwargs["query_struct"]
                call = {
                    "target": query_struct["target"],
                    "hidden_marker": float(kwargs["beit3_hidden"][0, 0, 0].item()),
                    "tokens": list(kwargs["tokens_list"]),
                }
                self.calls.append(call)
                return QueryGraph(
                    nodes=[torch.tensor([call["hidden_marker"]])] * 4,
                    edges=torch.tensor([[0], [1]], dtype=torch.long),
                    node_types=["category", "attribute", "relation", "action"],
                )

        parser = _Parser()
        model = SimpleNamespace(
            vr_ov_query_parser=parser,
            tokenizer=_Tokenizer(),
            _prompt_payloads_for_image=OpenWorldSAM2._prompt_payloads_for_image,
            _validate_query_graph=OpenWorldSAM2._validate_query_graph,
        )
        batched_inputs = [
            {
                "prompt": ["raw first", "raw second"],
                "query_text": ["raw first", "raw second"],
                "query_struct": [
                    {"target": "dog", "attributes": [], "relations": [], "actions": [], "negatives": [], "exists": True},
                    {"target": "cat", "attributes": ["red"], "relations": [], "actions": [], "negatives": [], "exists": True},
                ],
                "composed_prompt": ["legacy dog prompt", "legacy cat prompt"],
            },
            {
                "prompt": ["raw third"],
                "query_text": ["raw third"],
                "query_struct": [
                    {"target": "bird", "attributes": [], "relations": [], "actions": [], "negatives": [], "exists": True}
                ],
                "composed_prompt": ["legacy bird prompt"],
            },
        ]
        encoder_out = torch.tensor(
            [
                [[1.0, 0.0], [1.1, 0.0], [1.2, 0.0]],
                [[2.0, 0.0], [2.1, 0.0], [2.2, 0.0]],
                [[3.0, 0.0], [3.1, 0.0], [3.2, 0.0]],
            ]
        )
        input_ids = torch.tensor([[11, 12, 0], [21, 22, 0], [31, 0, 0]])
        attention_masks = torch.tensor(
            [[True, True, False], [True, True, False], [True, False, False]]
        )

        parsed = OpenWorldSAM2._run_vr_ov_query_parser(
            model,
            batched_inputs=batched_inputs,
            offset=[0, 2, 3],
            output={"encoder_out": encoder_out},
            input_ids=input_ids,
            attention_masks=attention_masks,
        )

        assert len(parsed) == 2
        assert [len(image_graphs) for image_graphs in parsed] == [2, 1]
        assert all(isinstance(graph, QueryGraph) for image_graphs in parsed for graph in image_graphs)
        assert [call["target"] for call in parser.calls] == ["dog", "cat", "bird"]

    def test_open_world_sam2_legacy_prompt_selection_keeps_flattened_compatibility(
        self,
    ) -> None:
        model = SimpleNamespace(
            learned_parser_enabled=False,
            reasonseg_enabled=True,
            composition_mode="composed_prompt",
        )

        prompts = OpenWorldSAM2._select_prompts(
            model,
            {
                "prompt": ["raw prompt"],
                "query_text": ["raw prompt"],
                "query_struct": [
                    {"target": "dog", "attributes": [], "relations": [], "actions": [], "negatives": [], "exists": True}
                ],
                "composed_prompt": ["legacy flattened prompt"],
            },
        )

        assert prompts == ["legacy flattened prompt"]

    def test_vr_ov_forward_routes_scene_graph_and_composition_modules(self) -> None:
        seen: dict[str, float] = {}

        def _decode_masks_for_image(*, features, batch_feat_with_tokens, img_idx):
            seen["token_marker"] = float(batch_feat_with_tokens[0, 0, 0].item())
            return torch.zeros(2, 1, 2, 2), torch.ones(2, 1), [torch.ones(1, 256, 2, 2)]

        def _build_processed_result(**kwargs):
            comp_scores = kwargs["comp_scores"]
            seen["comp_marker"] = float(comp_scores.attr_feat[0, 0, 0, 0].item())
            return {"comp_marker": seen["comp_marker"], "token_marker": seen["token_marker"]}

        model = SimpleNamespace(
            training=False,
            num_tokens=2,
            _assert_forward_backend_available=lambda: None,
            _prepare_input_tensors=lambda batched_inputs: (
                torch.zeros(1, 3, 2, 2),
                torch.zeros(1, 3, 2, 2),
                [(2, 2)],
            ),
            _encode_backbone_features=lambda images, batch_size: (
                {"backbone_fpn": [torch.ones(1, 256, 2, 2), torch.ones(1, 256, 2, 2)]},
                {"image_embed": torch.ones(1, 256, 2, 2), "high_res_feats": [torch.ones(1, 256, 2, 2)]},
            ),
            _run_vr_ov_scene_graph=lambda backbone_out: (
                torch.full((1, 5, 256), 3.0),
                torch.ones(1, 4, 256),
                torch.ones(1, 4, 50),
            ),
            _encode_text_prompts=lambda batched_inputs, images_evf: (
                (torch.zeros(1, 1, 256),),
                [0, 1],
                torch.ones(1, 2, dtype=torch.long),
                torch.ones(1, 2, dtype=torch.bool),
                {"encoder_out": torch.ones(1, 2, 256)},
            ),
            _run_vr_ov_query_parser=lambda **kwargs: [[{"query": "graph"}]],
            _update_learned_parser_logits=lambda output: None,
            _build_prompt_tokens=lambda prompt_feat: prompt_feat,
            _apply_vr_ov_scene_graph_prompt_context=OpenWorldSAM2._apply_vr_ov_scene_graph_prompt_context,
            _run_vr_ov_comp_matcher=lambda **kwargs: CompositionScores(
                cat_feat=torch.zeros(1, 1, 2, 2),
                attr_feat=torch.full((1, 1, 2, 2), 5.0),
                rel_feat=torch.zeros(1, 1, 2, 2),
                act_feat=torch.zeros(1, 1, 2, 2),
            ),
            _apply_cross_attention_prompt_context=lambda batch_feat_with_tokens, image_embed: batch_feat_with_tokens,
            _decode_masks_for_image=_decode_masks_for_image,
            _class_labels_for_image=lambda **kwargs: torch.zeros(2, dtype=torch.int64),
            _build_processed_result=_build_processed_result,
        )

        outputs = VR_OV.forward(
            model,
            [{"unique_categories": [0], "prompt": ["dog"], "query_text": ["dog"], "instances": []}],
        )

        assert outputs == [{"comp_marker": 5.0, "token_marker": 3.0}]
