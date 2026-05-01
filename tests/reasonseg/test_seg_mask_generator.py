from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from model.BIOtagging.bio_schema import NormalizedQuery, structure_to_bio_tags, tokens_to_bio_labels
from model.BIOtagging.complexity_selection import classify_tier, score_complexity, select_top_complex
from model.BIOtagging.seg_mask_generator import (
    _parse_key,
    _category_clean,
    _compute_centre,
    _generate_size_attrs,
    _generate_position_attrs,
    _spatial_relation,
    _make_rel_query,
    _q,
    generate_pairs_from_seg_mask,
    export_expanded_silver,
)


class TestKeyParsing:
    def test_simple_key(self) -> None:
        inst = _parse_key("000000391895-0-motorcycle.jpg")
        assert inst.image_id == "000000391895"
        assert inst.instance_idx == 0
        assert inst.category == "motorcycle"
        assert inst.target == "motorcycle"

    def test_multiword_category(self) -> None:
        inst = _parse_key("000000123456-3-baseball-bat.jpg")
        assert inst.category == "baseball bat"
        assert inst.target == "baseball bat"

    def test_underscore_category(self) -> None:
        assert _category_clean("cell_phone") == "cell phone"
        assert _category_clean("fire_hydrant") == "fire hydrant"

    def test_invalid_key_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_key("bogus.jpg")


class TestGeometryHeuristics:
    def test_compute_centre(self) -> None:
        cx, cy = _compute_centre([10.0, 20.0, 100.0, 200.0])
        assert cx == pytest.approx(60.0)
        assert cy == pytest.approx(120.0)

    def test_size_attrs_large(self) -> None:
        """Bbox covering >30% of image → 'large'"""
        attrs = _generate_size_attrs([0, 0, 300, 300], [500, 500])
        assert "large" in attrs

    def test_size_attrs_small(self) -> None:
        """Bbox covering ~4% → 'small'"""
        attrs = _generate_size_attrs([0, 0, 100, 100], [500, 500])
        assert "small" in attrs

    def test_size_attrs_tiny(self) -> None:
        """Bbox covering <2% → 'tiny'"""
        attrs = _generate_size_attrs([0, 0, 5, 5], [500, 500])
        assert "tiny" in attrs

    def test_size_attrs_tall(self) -> None:
        """Tall aspect ratio → 'tall'"""
        attrs = _generate_size_attrs([0, 0, 50, 200], [500, 500])
        assert "tall" in attrs

    def test_size_attrs_wide(self) -> None:
        """Wide aspect ratio → 'wide'"""
        attrs = _generate_size_attrs([0, 0, 300, 50], [500, 500])
        assert "wide" in attrs

    def test_size_attrs_none_for_mid_range(self) -> None:
        """Mid-size, mid-aspect (square) bbox → no attrs"""
        attrs = _generate_size_attrs([50, 50, 150, 150], [500, 500])
        assert attrs == []

    def test_position_attrs_corner(self) -> None:
        attrs = _generate_position_attrs([0, 0, 50, 50], [500, 500])
        assert any("top" in a and "left" in a for a in attrs)

    def test_position_attrs_center_returns_empty(self) -> None:
        attrs = _generate_position_attrs([200, 200, 100, 100], [500, 500])
        assert attrs == []

    def test_position_attrs_right_edge(self) -> None:
        attrs = _generate_position_attrs([400, 200, 80, 100], [500, 500])
        assert attrs == ["right"]


class TestSpatialRelation:
    def _inst(self, image_id: str, idx: int, cat: str, bbox: list[float]) -> object:
        from model.BIOtagging.seg_mask_generator import _Instance
        return _Instance(
            image_id=image_id, instance_idx=idx, category=cat,
            bbox=bbox, image_dim=[640, 480], key=f"{image_id}-{idx}-{cat}.jpg",
        )

    def test_horizontal_left(self) -> None:
        subj = self._inst("img1", 0, "dog", [100, 100, 50, 50])
        obj = self._inst("img1", 1, "person", [300, 100, 50, 50])
        rel = _spatial_relation(subj, obj)
        assert rel is not None
        assert rel == ("right of", "person")

    def test_horizontal_right(self) -> None:
        subj = self._inst("img1", 0, "person", [300, 100, 50, 50])
        obj = self._inst("img1", 1, "dog", [100, 100, 50, 50])
        rel = _spatial_relation(subj, obj)
        assert rel is not None
        assert rel == ("left of", "dog")

    def test_vertical_above(self) -> None:
        subj = self._inst("img1", 0, "bird", [200, 50, 50, 50])
        obj = self._inst("img1", 1, "tree", [200, 300, 100, 200])
        rel = _spatial_relation(subj, obj)
        assert rel is not None
        assert rel == ("below", "tree")

    def test_vertical_below(self) -> None:
        subj = self._inst("img1", 0, "cup", [200, 400, 50, 50])
        obj = self._inst("img1", 1, "table", [200, 100, 200, 100])
        rel = _spatial_relation(subj, obj)
        assert rel is not None
        assert rel == ("above", "table")

    def test_different_images_returns_none(self) -> None:
        subj = self._inst("img1", 0, "dog", [100, 100, 50, 50])
        obj = self._inst("img2", 0, "cat", [100, 100, 50, 50])
        assert _spatial_relation(subj, obj) is None

    def test_same_instance_returns_none(self) -> None:
        inst = self._inst("img1", 0, "dog", [100, 100, 50, 50])
        assert _spatial_relation(inst, inst) is None

    def test_near_overlap_returns_none(self) -> None:
        subj = self._inst("img1", 0, "dog", [100, 100, 50, 50])
        obj = self._inst("img1", 1, "cat", [100, 101, 50, 50])
        assert _spatial_relation(subj, obj) is None

    def test_diagonal_relation(self) -> None:
        subj = self._inst("img1", 0, "motorcycle", [50, 50, 100, 100])
        obj = self._inst("img1", 1, "person", [200, 200, 80, 120])
        rel = _spatial_relation(subj, obj)
        assert rel is not None
        rel_word, rel_tgt = rel
        assert "below" in rel_word
        assert "right of" in rel_word
        assert "and" in rel_word
        assert rel_tgt == "person"


class TestMakeRelQuery:
    def _inst(self, image_id: str, idx: int, cat: str, bbox: list[float]) -> object:
        from model.BIOtagging.seg_mask_generator import _Instance
        return _Instance(
            image_id=image_id, instance_idx=idx, category=cat,
            bbox=bbox, image_dim=[640, 480], key=f"{image_id}-{idx}-{cat}.jpg",
        )

    def test_basic_rel_query(self) -> None:
        subj = self._inst("img1", 0, "motorcycle", [100, 200, 80, 60])
        obj = self._inst("img1", 1, "person", [300, 200, 50, 100])
        result = _make_rel_query(subj, obj)
        assert result is not None
        query, structure = result
        assert query == "motorcycle right of person"
        assert structure["target"] == "motorcycle"
        assert structure["relations"] == [{"type": "right of", "target": "person"}]
        assert structure["exists"] is True

    def test_rel_query_projects_to_non_o_labels(self) -> None:
        subj = self._inst("img1", 0, "dog", [100, 100, 60, 60])
        obj = self._inst("img1", 1, "bicycle", [300, 100, 80, 80])
        result = _make_rel_query(subj, obj)
        assert result is not None
        query, structure = result
        tokens = query.lower().split()
        labels = tokens_to_bio_labels(tokens, structure)
        non_o = sum(1 for l in labels if l != 0)
        assert non_o >= 3  # subject, relation word, rel-target


class TestGeneratorOutputShape:
    @pytest.fixture(scope="class")
    def _mini_seg_mask(self) -> Path:
        """Create a tiny JSON with 2 images, 3 instances."""
        import os
        data = {
            "000000000001-0-dog.jpg": {
                "segmentation": [[10, 10, 60, 10, 60, 60, 10, 60]],
                "bbox": [10, 10, 50, 50],
                "image_dim": [200, 300],
            },
            "000000000001-1-cat.jpg": {
                "segmentation": [[150, 150, 250, 150, 250, 190, 150, 190]],
                "bbox": [150, 150, 100, 40],
                "image_dim": [200, 300],
            },
            "000000000002-0-motorcycle.jpg": {
                "segmentation": [[50, 80, 200, 80, 200, 180, 50, 180]],
                "bbox": [50, 80, 150, 100],
                "image_dim": [300, 500],
            },
        }
        tmp = Path(tempfile.mkdtemp()) / "mini_seg_mask.json"
        tmp.write_text(json.dumps(data))
        return tmp

    def test_output_is_list_of_pairs(self, _mini_seg_mask: Path) -> None:
        pairs = generate_pairs_from_seg_mask(_mini_seg_mask, seed=7, max_noun=0)
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        for q, s in pairs:
            assert isinstance(q, str)
            assert isinstance(s, dict)
            assert "target" in s
            assert "attributes" in s
            assert "relations" in s
            assert "actions" in s
            assert "negatives" in s
            assert "exists" in s

    def test_contains_relation_example(self, _mini_seg_mask: Path) -> None:
        pairs = generate_pairs_from_seg_mask(_mini_seg_mask, seed=7, max_noun=0)
        tiers = [classify_tier(s) for _, s in pairs]
        assert "relation" in tiers, f"Expected at least one relation query, got tiers: {tiers}"

    def test_all_structures_parse_to_bio_tags(self, _mini_seg_mask: Path) -> None:
        pairs = generate_pairs_from_seg_mask(_mini_seg_mask, seed=7, max_noun=0)
        for query, structure in pairs:
            tokens = query.lower().split()
            tags = structure_to_bio_tags(structure, tokens)
            assert len(tags) == len(tokens)
            # Every token must map to a valid BIO tag
            from model.BIOtagging.bio_schema import BIO_TAGS
            for tag in tags:
                assert tag in BIO_TAGS, f"Invalid tag {tag!r} for query {query!r}"


class TestExportExpandedSilver:
    def test_export_format(self) -> None:
        pairs: list[tuple[str, NormalizedQuery]] = [
            ("dog behind bicycle", _q(
                target="dog",
                relations=[{"type": "behind", "target": "bicycle"}],
            )),
            ("large motorcycle", _q(
                target="motorcycle",
                attributes=["large"],
            )),
        ]
        tmp = Path(tempfile.mkdtemp()) / "silver.json"
        path = export_expanded_silver(pairs, tmp)
        assert path == tmp
        assert tmp.exists()
        loaded = json.loads(tmp.read_text())
        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert loaded[0] == ["dog behind bicycle", {
            "target": "dog",
            "attributes": [],
            "relations": [{"type": "behind", "target": "bicycle"}],
            "actions": [],
            "negatives": [],
            "exists": True,
        }]
        assert loaded[1] == ["large motorcycle", {
            "target": "motorcycle",
            "attributes": ["large"],
            "relations": [],
            "actions": [],
            "negatives": [],
            "exists": True,
        }]

    def test_export_with_top_k(self) -> None:
        pairs: list[tuple[str, NormalizedQuery]] = [
            ("dog", _q(target="dog")),
            ("dog behind bicycle", _q(
                target="dog",
                relations=[{"type": "behind", "target": "bicycle"}],
            )),
            ("red dress", _q(target="dress", attributes=["red"])),
        ]
        tmp = Path(tempfile.mkdtemp()) / "silver_topk.json"
        export_expanded_silver(pairs, tmp, top_k=2)
        loaded = json.loads(tmp.read_text())
        assert len(loaded) == 2
        tiers = [classify_tier(item[1]) for item in loaded]
        assert "relation" in tiers

    def test_top_k_larger_than_input(self) -> None:
        pairs: list[tuple[str, NormalizedQuery]] = [
            ("dog", _q(target="dog")),
        ]
        tmp = Path(tempfile.mkdtemp()) / "silver_overflow.json"
        export_expanded_silver(pairs, tmp, top_k=10)
        loaded = json.loads(tmp.read_text())
        assert len(loaded) == 1


class TestComplexitySelectionOnGenerated:
    """Verify generated relation queries score higher than attribute/noun."""

    def test_relation_scores_higher_than_attr(self) -> None:
        rel_pair = ("dog left of cat", _q(
            target="dog",
            relations=[{"type": "left of", "target": "cat"}],
        ))
        attr_pair = ("small dog", _q(
            target="dog",
            attributes=["small"],
        ))
        assert score_complexity(*rel_pair) > score_complexity(*attr_pair)

    def test_relation_scores_higher_than_noun(self) -> None:
        rel_pair = ("dog left of cat", _q(
            target="dog",
            relations=[{"type": "left of", "target": "cat"}],
        ))
        noun_pair = ("dog", _q(target="dog"))
        assert score_complexity(*rel_pair) > score_complexity(*noun_pair)

    def test_rel_attr_scores_above_relation(self) -> None:
        rel_attr_pair = ("large dog left of cat", _q(
            target="dog",
            attributes=["large"],
            relations=[{"type": "left of", "target": "cat"}],
        ))
        rel_pair = ("dog left of cat", _q(
            target="dog",
            relations=[{"type": "left of", "target": "cat"}],
        ))
        assert score_complexity(*rel_attr_pair) > score_complexity(*rel_pair)
