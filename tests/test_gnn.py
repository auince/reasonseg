from __future__ import annotations

import pytest
import torch

from model.gnn import QueryGraphGAT


class TestQueryGraphGAT:
    """Unit tests for the QueryGraphGAT graph reasoning module."""

    @staticmethod
    def _make_model(**kwargs: int | float) -> QueryGraphGAT:
        return QueryGraphGAT(**kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # test_forward_shape
    # ------------------------------------------------------------------ #
    def test_forward_shape(self) -> None:
        """4-node graph forward pass → output [4, 128]."""
        model = self._make_model()
        nodes = torch.randn(4, 768)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        output = model(nodes, edges)
        assert output.shape == (4, 128)

    # ------------------------------------------------------------------ #
    # test_single_node
    # ------------------------------------------------------------------ #
    def test_single_node(self) -> None:
        """Single-node graph → output [1, 128] with no NaN."""
        model = self._make_model()
        nodes = torch.randn(1, 768)
        edges = torch.tensor([[], []], dtype=torch.long)

        output = model(nodes, edges)
        assert output.shape == (1, 128)
        assert not torch.isnan(output).any()

    # ------------------------------------------------------------------ #
    # test_empty_edges
    # ------------------------------------------------------------------ #
    def test_empty_edges(self) -> None:
        """Graph with no edges (only self-loops) → does not crash."""
        model = self._make_model()
        nodes = torch.randn(5, 768)
        edges = torch.tensor([[], []], dtype=torch.long)

        output = model(nodes, edges)
        assert output.shape == (5, 128)
        assert not torch.isnan(output).any()

    # ------------------------------------------------------------------ #
    # test_different_dims
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        ("in_dim", "hidden_dim", "out_dim", "heads"),
        [
            (512, 128, 64, 2),
            (768, 256, 128, 4),
            (1024, 512, 256, 8),
        ],
    )
    def test_different_dims(
        self, in_dim: int, hidden_dim: int, out_dim: int, heads: int,
    ) -> None:
        """Custom dimensions → output matches out_dim."""
        model = QueryGraphGAT(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, heads=heads,
        )
        nodes = torch.randn(6, in_dim)
        edges = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)

        output = model(nodes, edges)
        assert output.shape == (6, out_dim)
        assert not torch.isnan(output).any()

    # ------------------------------------------------------------------ #
    # test_deterministic
    # ------------------------------------------------------------------ #
    def test_deterministic(self) -> None:
        """Same input → same output in eval mode."""
        model = self._make_model()
        model.eval()

        torch.manual_seed(42)
        nodes = torch.randn(4, 768)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        out1 = model(nodes, edges)

        torch.manual_seed(42)
        nodes2 = torch.randn(4, 768)
        out2 = model(nodes2, edges)

        assert torch.allclose(out1, out2, atol=1e-6)
