"""Official MARIDA splits and scene-grouped cross-validation folds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .ingestion import MaridaDataset


def load_split(path: Path) -> list[str]:
    """Parse one split file (train_X.txt, val_X.txt, test_X.txt) into patch ids."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    return lines


def load_official_splits(splits_dir: Path) -> dict[str, list[str]]:
    """Return {'train': [...], 'val': [...], 'test': [...]}."""
    return {
        name: load_split(splits_dir / f"{name}_X.txt")
        for name in ("train", "val", "test")
    }


def assert_scene_disjoint(splits: dict[str, list[str]], dataset: MaridaDataset) -> None:
    """Raise AssertionError if any scene appears in more than one split."""
    scenes = {}
    for split_name, patch_ids in splits.items():
        for patch_id in patch_ids:
            scenes.setdefault(dataset.read_scene_id(patch_id), split_name)
            if scenes[dataset.read_scene_id(patch_id)] != split_name:
                raise AssertionError(
                    f"scene {dataset.read_scene_id(patch_id)} leaks across splits"
                )


def scene_grouped_folds(
    patch_ids: list[str],
    dataset: MaridaDataset,
    n_folds: int = 10,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified-by-scene fold generator using GroupKFold on scene ids.

    Returns a list of (train_indices, val_indices) over the input list, where
    no scene appears in both folds of a pair.
    """
    groups = np.array([dataset.read_scene_id(pid) for pid in patch_ids])
    generator = GroupKFold(n_splits=n_folds)
    return [
        (train_idx, val_idx)
        for train_idx, val_idx in generator.split(patch_ids, groups=groups)
    ]