"""Tests for Flaw 1.5: teacher pseudo-label data-leakage audit (manifest + pre-flight).

Covers:
  * scripts/train_one.py   -> write_train_manifest / reaudit_teacher_manifest
  * scripts/train_gan_augmented.py -> pre-flight teacher leak verification
  * calibration split-purity guards
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_gan_augmented import (  # noqa: E402
    load_teacher_manifest,
    verify_teacher_no_leakage,
)
from scripts.train_one import (  # noqa: E402
    _sha256_file,
    reaudit_teacher_manifest,
    write_train_manifest,
)

MANIFEST = "train_manifest.json"


def _make_splits(tmp_path: Path) -> tuple[Path, dict[str, list[str]]]:
    """Create a tiny synthetic split pool: 6 train, 2 val, 2 test ids."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    split_ids = {
        "train": [f"t_{i}" for i in range(6)],
        "val": [f"v_{i}" for i in range(2)],
        "test": [f"e_{i}" for i in range(2)],
    }
    for name, ids in split_ids.items():
        (splits_dir / f"{name}_X.txt").write_text("\n".join(ids) + "\n")
    return splits_dir, split_ids


class TestWriteTrainManifest:
    def test_writes_manifest_with_disjoint_contract(self, tmp_path):
        splits_dir, split_ids = _make_splits(tmp_path)
        out_dir = tmp_path / "artifacts" / "baseline"
        manifest = write_train_manifest(
            "baseline", out_dir, splits_dir,
            split_ids["train"], split_ids["val"], split_ids["test"],
        )
        written = json.loads((out_dir / MANIFEST).read_text())
        assert written == manifest
        assert manifest["variant"] == "baseline"
        assert manifest["train_val_disjoint"] is True
        assert manifest["train_test_disjoint"] is True
        assert manifest["n_train"] == 6
        assert set(manifest["train_patch_ids"]) == set(split_ids["train"])
        assert set(manifest["test_patch_ids"]) == set(split_ids["test"])
        assert set(manifest["train_patch_ids"]).isdisjoint(set(manifest["val_patch_ids"]))
        assert set(manifest["train_patch_ids"]).isdisjoint(set(manifest["test_patch_ids"]))

    def test_split_hashes_are_sha256_of_files(self, tmp_path):
        splits_dir, _ = _make_splits(tmp_path)
        out_dir = tmp_path / "artifacts" / "baseline"
        manifest = write_train_manifest(
            "baseline", out_dir, splits_dir,
            ["t_0", "t_1"], ["v_0"], ["e_0"],
        )
        for name, path in (("train", "train_X.txt"), ("val", "val_X.txt"), ("test", "test_X.txt")):
            assert manifest["split_hashes_sha256"][name] == _sha256_file(splits_dir / path)

    def test_rejects_leak_into_val(self, tmp_path):
        splits_dir, split_ids = _make_splits(tmp_path)
        out_dir = tmp_path / "artifacts" / "baseline"
        with pytest.raises(RuntimeError, match="Refusing to write train manifest"):
            write_train_manifest(
                "baseline", out_dir, splits_dir,
                split_ids["train"] + [split_ids["val"][0]],
                split_ids["val"], split_ids["test"],
            )
        assert not (out_dir / MANIFEST).exists()

    def test_rejects_leak_into_test(self, tmp_path):
        splits_dir, split_ids = _make_splits(tmp_path)
        out_dir = tmp_path / "artifacts" / "baseline"
        with pytest.raises(RuntimeError, match="Refusing to write train manifest"):
            write_train_manifest(
                "baseline", out_dir, splits_dir,
                split_ids["train"] + [split_ids["test"][0]],
                split_ids["val"], split_ids["test"],
            )

    @pytest.mark.requires_data
    def test_reaudit_teacher_manifest_over_official_splits(self):
        manifest = reaudit_teacher_manifest("baseline")
        assert manifest["variant"] == "baseline"
        assert manifest["source"] == "reaudit_official_splits"
        assert manifest["train_val_disjoint"] is True
        assert manifest["train_test_disjoint"] is True
        assert manifest["n_train"] == 694
        assert manifest["n_val"] == 328
        assert manifest["n_test"] == 359


class TestTeacherLeakVerification:
    def _write_teacher_manifest(self, artifacts_dir: Path, train_ids: list[str]) -> None:
        teacher_dir = artifacts_dir / "teacher"
        teacher_dir.mkdir(parents=True, exist_ok=True)
        (teacher_dir / MANIFEST).write_text(json.dumps({"train_patch_ids": train_ids}))

    def test_clean_teacher_passes(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        self._write_teacher_manifest(artifacts, ["t_0", "t_1", "t_2"])
        splits = {"train": ["t_0"], "val": ["v_0"], "test": ["e_0"]}
        result = verify_teacher_no_leakage("teacher", splits, artifacts)
        assert result["teacher_leakage_verified"] is True
        assert result["n_val_overlap"] == 0 and result["n_test_overlap"] == 0

    def test_raises_on_val_overlap(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        self._write_teacher_manifest(artifacts, ["t_0", "v_0"])
        splits = {"train": ["t_0"], "val": ["v_0"], "test": ["e_0"]}
        with pytest.raises(RuntimeError, match="Potential data leakage detected in teacher model!"):
            verify_teacher_no_leakage("teacher", splits, artifacts)

    def test_raises_on_test_overlap(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        self._write_teacher_manifest(artifacts, ["t_0", "e_0"])
        splits = {"train": ["t_0"], "val": ["v_0"], "test": ["e_0"]}
        with pytest.raises(RuntimeError, match="Potential data leakage detected in teacher model!"):
            verify_teacher_no_leakage("teacher", splits, artifacts)

    def test_missing_manifest_raises(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        splits = {"train": ["t_0"], "val": ["v_0"], "test": ["e_0"]}
        with pytest.raises(RuntimeError, match="Teacher contract manifest missing"):
            verify_teacher_no_leakage("teacher", splits, artifacts)

    def test_load_teacher_manifest_reads_json(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        self._write_teacher_manifest(artifacts, ["t_0"])
        assert load_teacher_manifest("teacher", artifacts)["train_patch_ids"] == ["t_0"]


class TestCalibrationPurityGuard:
    """The threshold calibration must sample ONLY from splits['train']."""

    def _splits(self):
        return {
            "train": [f"t_{i}" for i in range(12)],
            "val": [f"v_{i}" for i in range(4)],
            "test": [f"e_{i}" for i in range(4)],
        }

    def test_train_slice_is_subset_and_disjoint(self):
        splits = self._splits()
        sample_ids = splits["train"][:8]
        assert set(sample_ids).issubset(set(splits["train"])), "Calibration set leaked into non-train splits!"
        assert set(sample_ids).isdisjoint(set(splits["val"])), "Validation leak in calibration!"
        assert set(sample_ids).isdisjoint(set(splits["test"])), "Test leak in calibration!"

    def test_a_val_id_in_calibration_trips_assert(self):
        splits = self._splits()
        sample_ids = splits["train"][:3] + [splits["val"][0]]
        with pytest.raises(AssertionError):
            assert set(sample_ids).isdisjoint(set(splits["val"])), "Validation leak in calibration!"