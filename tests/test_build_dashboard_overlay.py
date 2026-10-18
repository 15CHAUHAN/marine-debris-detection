"""Tests for the ground-truth overlay helpers in scripts/build_dashboard.py."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_dashboard import _draw_debris_boxes, _gt_overlay  # noqa: E402


def _rgb(n: int = 10) -> np.ndarray:
    # distinct per-pixel color so blending with a class/grey never aliases.
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    return np.stack([x / n, y / n, np.zeros((n, n), dtype=np.float64)], axis=-1)


def test_overlay_dimmed_rgb_for_unlabeled():
    rgb = _rgb()
    mask = np.zeros((10, 10), dtype=np.int64)
    out = _gt_overlay(rgb, mask)
    assert out.shape == rgb.shape
    np.testing.assert_allclose(out, 0.7 * rgb, atol=1e-12)


def test_overlay_tints_labeled_pixels_with_class_color():
    rgb = _rgb()
    mask = np.zeros((10, 10), dtype=np.int64)
    mask[5, 5] = 6  # Marine Water -> CLASS_COLORS[6] (grey) blend still differs from 0.7*rgb
    mask[0, 0] = 2  # Dense Sargassum
    out = _gt_overlay(rgb, mask)
    assert float(out[0, 0, 0]) > float(0.7 * rgb[0, 0, 0])
    # unlabeled area stays the dimmed satellite rgb
    np.testing.assert_allclose(out[9, 9], 0.7 * rgb[9, 9], atol=1e-12)


def test_overlay_never_exceeds_unit_range():
    rng = np.random.default_rng(0)
    rgb = rng.random((16, 16, 3))
    mask = rng.integers(0, 12, size=(16, 16)).astype(np.int64)
    out = _gt_overlay(rgb, mask)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_debris_boxes_drawn_for_single_pixel_debris():
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.invert_yaxis()
    mask = np.zeros((10, 10), dtype=np.int64)
    mask[4, 4] = 1
    _draw_debris_boxes(ax, mask)
    patches = [p for p in ax.patches if isinstance(p, plt.Rectangle)]
    texts = ax.texts
    plt.close(fig)
    assert len(patches) == 1
    assert len(texts) == 1
    assert texts[0].get_text() == "debris"
    rect = patches[0]
    assert rect.get_edgecolor()[0] == 1.0  # yellow channel
    # dilated box must cover the debris pixel
    xmin, ymin = rect.get_xy()
    assert xmin <= 4 <= xmin + rect.get_width()
    assert ymin <= 4 <= ymin + rect.get_height()


def test_no_boxes_without_debris():
    fig, ax = plt.subplots(figsize=(2, 2))
    mask = np.zeros((10, 10), dtype=np.int64)
    _draw_debris_boxes(ax, mask)
    plt.close(fig)
    assert len(ax.patches) == 0
    assert len(ax.texts) == 0