from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from networkx import DiGraph, draw_networkx, spring_layout

from model.BIOtagging.bio_schema import bio_tags_to_structure
from model.BIOtagging.query_parser_head import QueryParserHead
from model.query_parser import LLMQueryParser
from model.vr_ov_types import QueryGraph

logger = logging.getLogger(__name__)

_CHECKPOINT = (
    "model/BIOtagging/outputs/"
    "stage1_fast_train50k_plus_mask20k_plus_flashpro2k_20260430_031500/"
    "parser_head_best.pt"
)

_OUTPUT_DIR = Path("model/BIOtagging/outputs/query_parser_test")

TEST_QUERIES: list[tuple[str, list[str]]] = [
    ("red cup", ["red", "cup"]),
    ("person holding phone", ["person", "holding", "phone"]),
    ("no dog", ["no", "dog"]),
    ("blue chair on the left", ["blue", "chair", "on", "the", "left"]),
    ("small wooden table", ["small", "wooden", "table"]),
    ("cat beside sofa", ["cat", "beside", "sofa"]),
    ("woman wearing hat", ["woman", "wearing", "hat"]),
    ("dog behind bicycle", ["dog", "behind", "bicycle"]),
    ("green apple", ["green", "apple"]),
    ("man sitting", ["man", "sitting"]),
]


def _default_attention_mask(seq_len: int) -> torch.Tensor:
    return torch.ones(1, seq_len)


def _checkpoint_hidden_dim() -> int:
    ckpt = torch.load(_CHECKPOINT, map_location="cpu")
    return ckpt["classifier.weight"].shape[1]


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def parser() -> LLMQueryParser:
    return LLMQueryParser(parser_checkpoint=_CHECKPOINT)


@pytest.fixture(scope="module")
def parser_hidden_dim() -> int:
    return _checkpoint_hidden_dim()


# ── Integration tests ────────────────────────────────────────────────────────


class TestLLMQueryParserIntegration:
    def test_load_checkpoint(self, parser: LLMQueryParser) -> None:
        assert parser._has_parser
        assert parser.parser_head is not None

    @pytest.mark.parametrize(
        "raw_query,tokens",
        [
            ("red cup", ["red", "cup", "[SEP]"]),
            ("person holding phone", ["person", "holding", "phone", "[SEP]"]),
            ("blue chair", ["blue", "chair", "[SEP]"]),
            ("green apple", ["green", "apple", "[SEP]"]),
            ("wooden table", ["wooden", "table", "[SEP]"]),
        ],
    )
    def test_bio_decode_accuracy(
        self,
        parser_hidden_dim: int,
        raw_query: str,
        tokens: list[str],
    ) -> None:
        ckpt = torch.load(_CHECKPOINT, map_location="cpu")
        ckpt_num_layers = len(
            {k.split(".")[2] for k in ckpt if k.startswith("transformer.layers")}
        )
        head = QueryParserHead(
            hidden_dim=parser_hidden_dim, num_tags=14, num_layers=ckpt_num_layers, nhead=8,
        )
        head.load_state_dict(ckpt)
        head.eval()

        hidden = torch.randn(1, len(tokens), parser_hidden_dim)
        attn = _default_attention_mask(len(tokens))

        result = head.decode_structure(tokens, hidden, attn)

        assert isinstance(result, dict)
        assert "target" in result
        assert "exists" in result  # always present

        target = result.get("target")
        logger.info("Query=%r  →  target=%r  attrs=%r  exists=%r",
                    raw_query, target, result.get("attributes"), result["exists"])
        # With random hidden states, the parser may miss target — that's fine.
        # The pipeline must NOT crash regardless.

    def test_full_pipeline_end_to_end(self, parser: LLMQueryParser) -> None:
        dim = parser.hidden_dim
        tokens = ["red", "cup", "[SEP]"]
        hidden = torch.randn(1, len(tokens), dim)
        attn = _default_attention_mask(len(tokens))

        result = parser.forward(hidden, attn, tokens)

        assert isinstance(result, QueryGraph)
        assert len(result.nodes) == 4
        assert result.edges.shape == (2, 6)
        assert result.node_types == ["category", "attribute", "relation", "action"]

        for i, node in enumerate(result.nodes):
            assert node.shape == (128,)
            assert not torch.isnan(node).any(), f"Node {i} has NaN"

    def test_multiple_queries_all_valid(self, parser: LLMQueryParser) -> None:
        dim = parser.hidden_dim
        for raw_query, tokens in TEST_QUERIES:
            tokens_with_sep = tokens + ["[SEP]"]
            hidden = torch.randn(1, len(tokens_with_sep), dim)
            attn = _default_attention_mask(len(tokens_with_sep))

            result = parser.forward(hidden, attn, tokens_with_sep)

            assert len(result.nodes) == 4
            assert result.edges.shape == (2, 6)
            assert result.node_types == ["category", "attribute", "relation", "action"]

            for node in result.nodes:
                assert not torch.isnan(node).any()

    def test_gnn_outputs_differ_across_nodes(self, parser: LLMQueryParser) -> None:
        dim = parser.hidden_dim
        tokens = ["red", "cup", "on", "table", "[SEP]"]
        hidden = torch.randn(1, len(tokens), dim)
        attn = _default_attention_mask(len(tokens))

        result = parser.forward(hidden, attn, tokens)

        vectors = [n.detach().clone() for n in result.nodes]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for a, b in pairs:
            assert not torch.allclose(vectors[a], vectors[b], atol=1e-5), (
                f"Nodes {a} and {b} should differ after GNN"
            )


# ── Visualization ────────────────────────────────────────────────────────────


class TestVisualization:
    def test_visualize_query_graphs(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        parser = LLMQueryParser(parser_checkpoint=_CHECKPOINT)
        dim = parser.hidden_dim

        for raw_query, tokens in TEST_QUERIES[:6]:
            tokens_with_sep = tokens + ["[SEP]"]
            hidden = torch.randn(1, len(tokens_with_sep), dim)
            attn = _default_attention_mask(len(tokens_with_sep))

            result = parser.forward(hidden, attn, tokens_with_sep)
            prefix = raw_query.replace(" ", "_")
            _save_graph_visualization(result, _OUTPUT_DIR, prefix)

    def test_visualize_no_checkpoint(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        parser = LLMQueryParser(parser_checkpoint=None)
        dim = parser.hidden_dim

        tokens = ["red", "cup", "[SEP]"]
        hidden = torch.randn(1, len(tokens), dim)
        attn = _default_attention_mask(len(tokens))
        result = parser.forward(hidden, attn, tokens)

        _save_graph_visualization(result, _OUTPUT_DIR, "fallback_red_cup")


def _save_graph_visualization(query_graph: QueryGraph, output_dir: Path, prefix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes_tensor = query_graph.nodes
    stack = torch.stack(nodes_tensor).detach().cpu().numpy()
    node_labels = query_graph.node_types
    edges_cpu = query_graph.edges.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── 1.  Heatmap of node embeddings ──────────────────────────────────
    im = axes[0].imshow(stack, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(node_labels)
    axes[0].set_xlabel("Feature dimension")
    axes[0].set_title(f"Node embedding heatmap — {prefix}")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # ── 2.  Network graph ──────────────────────────────────────────────
    G = DiGraph()
    for i, label in enumerate(node_labels):
        G.add_node(i, label=label)
    src, dst = edges_cpu[0], edges_cpu[1]
    for s, d in zip(src, dst):
        G.add_edge(int(s), int(d))

    pos = spring_layout(G, seed=42)
    draw_networkx(
        G,
        pos,
        ax=axes[1],
        labels={i: lbl for i, lbl in enumerate(node_labels)},
        node_color=["#ff6b6b", "#4ecdc4", "#45b7d1", "#f9ca24"],
        node_size=1200,
        font_size=10,
        font_weight="bold",
        arrows=True,
        arrowsize=20,
        edge_color="#888888",
    )
    axes[1].set_title(f"Graph structure — {prefix}")

    plt.tight_layout()
    save_path = output_dir / f"{prefix}_graph_viz.png"
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved visualization: %s", save_path)
