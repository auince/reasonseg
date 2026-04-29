from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import pickle
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load_prepare_refcoco_main():
    module_path = ROOT / "reasonseg" / "cli_surface.py"
    spec = importlib.util.spec_from_file_location("reasonseg_cli_surface", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.prepare_refcoco_main


def _image(image_id: int) -> dict[str, Any]:
    return {
        "id": image_id,
        "file_name": f"COCO_train2014_{image_id:012d}.jpg",
        "height": 32,
        "width": 32,
        "license": 1,
    }


def _annotation(annotation_id: int, image_id: int) -> dict[str, Any]:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [1, 2, 3, 4],
        "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]],
        "area": 12,
        "iscrowd": 0,
    }


def _sentence(sent_id: int, raw: str) -> dict[str, Any]:
    return {
        "sent_id": sent_id,
        "sent": raw.lower(),
        "raw": raw,
        "tokens": raw.lower().split(),
    }


def _ref(
    ref_id: int,
    image_id: int,
    annotation_id: int,
    split: str,
    raw: str,
) -> dict[str, Any]:
    return {
        "ref_id": ref_id,
        "ann_id": annotation_id,
        "category_id": 1,
        "image_id": image_id,
        "file_name": f"COCO_train2014_{image_id:012d}_{ref_id}.jpg",
        "split": split,
        "sent_ids": [ref_id * 10],
        "sentences": [_sentence(ref_id * 10, raw)],
    }


def _write_dataset(
    root: Path, dataset_name: str, refs_file_name: str, refs: list[dict[str, Any]]
) -> None:
    dataset_root = root / dataset_name
    dataset_root.mkdir(parents=True)

    image_ids = sorted({int(ref["image_id"]) for ref in refs})
    annotation_ids = sorted({int(ref["ann_id"]) for ref in refs})
    image_by_id = {image_id: _image(image_id) for image_id in image_ids}
    annotation_by_id = {
        annotation_id: _annotation(annotation_id, image_ids[index])
        for index, annotation_id in enumerate(annotation_ids)
    }

    for ref in refs:
        annotation_by_id[int(ref["ann_id"])] = _annotation(
            int(ref["ann_id"]), int(ref["image_id"])
        )

    payload = {
        "images": [image_by_id[image_id] for image_id in image_ids],
        "annotations": [
            annotation_by_id[annotation_id] for annotation_id in annotation_ids
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    (dataset_root / "instances.json").write_text(json.dumps(payload), encoding="utf-8")
    with (dataset_root / refs_file_name).open("wb") as handle:
        pickle.dump(refs, handle)


def _build_dataset_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "datasets"
    (data_root / "coco" / "annotations").mkdir(parents=True)
    (data_root / "coco" / "train2014").mkdir(parents=True)
    (data_root / "coco" / "annotations" / "instances_train2014.json").write_text(
        json.dumps({"images": [], "annotations": [], "categories": []}),
        encoding="utf-8",
    )

    _write_dataset(
        data_root,
        "refcoco",
        "refs(unc).p",
        [
            _ref(1, 101, 1001, "train", "Train object"),
            _ref(2, 102, 1002, "val", "Val object"),
            _ref(3, 103, 1003, "testA", "Test A object"),
            _ref(4, 104, 1004, "testB", "Test B object"),
        ],
    )
    _write_dataset(
        data_root,
        "refcoco+",
        "refs(unc).p",
        [
            _ref(11, 201, 2001, "train", "Plus train object"),
            _ref(12, 202, 2002, "val", "Plus val object"),
            _ref(13, 203, 2003, "testA", "Plus test A object"),
            _ref(14, 204, 2004, "testB", "Plus test B object"),
        ],
    )
    _write_dataset(
        data_root,
        "refcocog",
        "refs(umd).p",
        [
            _ref(21, 301, 3001, "train", "Broccoli bowl"),
            _ref(22, 301, 3001, "train", "Wooden table under bowl"),
            _ref(23, 302, 3002, "val", "Val scene"),
            _ref(24, 303, 3003, "test", "Test scene"),
        ],
    )
    return data_root


def test_prepare_refcoco_materializes_and_verifies_outputs(
    tmp_path: Path, capsys
) -> None:
    prepare_refcoco_main = _load_prepare_refcoco_main()
    data_root = _build_dataset_root(tmp_path)

    exit_code = prepare_refcoco_main(
        ["--data-root", str(data_root), "--materialize", "--verify-only"]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "materialized=11, verified=11" in stdout

    output_path = data_root / "coco" / "annotations" / "refcocog_umd_train.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert set(payload) == {"images", "annotations"}
    assert len(payload["annotations"]) == 1
    annotation = payload["annotations"][0]
    assert annotation["bbox"] == [1, 2, 3, 4]
    assert annotation["segmentation"] == [[1, 2, 4, 2, 4, 6, 1, 6]]
    assert annotation["file_name"] == "COCO_train2014_000000000301.jpg"
    assert annotation["ref_ids"] == [21, 22]
    assert annotation["source_ref_count"] == 2
    assert (
        annotation["multi_ref_policy"] == "merge_same_ann_id_with_stable_sentence_union"
    )
    assert [sentence["raw"] for sentence in annotation["sentences"]] == [
        "Broccoli bowl",
        "Wooden table under bowl",
    ]

    verify_exit_code = prepare_refcoco_main(
        ["--data-root", str(data_root), "--verify-only"]
    )
    assert verify_exit_code == 0


def test_prepare_refcoco_reports_missing_raw_asset(tmp_path: Path, capsys) -> None:
    prepare_refcoco_main = _load_prepare_refcoco_main()
    data_root = _build_dataset_root(tmp_path)
    missing_path = data_root / "refcoco+" / "refs(unc).p"
    missing_path.unlink()

    exit_code = prepare_refcoco_main(["--data-root", str(data_root), "--verify-only"])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "Missing RefCOCO raw input asset(s):" in stderr
    assert str(missing_path) in stderr


def test_prepare_refcoco_reports_missing_data_root_requirements(
    tmp_path: Path, capsys
) -> None:
    prepare_refcoco_main = _load_prepare_refcoco_main()
    data_root = tmp_path / "missing-datasets"

    exit_code = prepare_refcoco_main(["--data-root", str(data_root), "--verify-only"])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "Missing RefCOCO raw input asset(s):" in stderr
    assert str(data_root / "refcoco" / "instances.json") in stderr
    assert str(data_root / "coco" / "train2014") in stderr


def test_prepare_refcoco_verify_only_reports_missing_materialized_output(
    tmp_path: Path, capsys
) -> None:
    prepare_refcoco_main = _load_prepare_refcoco_main()
    data_root = _build_dataset_root(tmp_path)
    output_path = data_root / "coco" / "annotations" / "refcoco_unc_val.json"

    assert prepare_refcoco_main(["--data-root", str(data_root), "--materialize"]) == 0
    output_path.unlink()

    exit_code = prepare_refcoco_main(["--data-root", str(data_root), "--verify-only"])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "Missing RefCOCO materialized output(s):" in stderr
    assert str(output_path) in stderr
