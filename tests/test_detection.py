"""Tests for src/detection/labels.py (YOLO debris bbox extraction)."""
import numpy as np

from src.detection.labels import DebrisBox, debris_boxes, rgb_composite, to_yolo_lines


def test_empty_mask_has_no_boxes():
    mask = np.zeros((256, 256), dtype=np.int64)
    assert debris_boxes(mask) == []


def test_single_debris_blob_bbox_extent():
    mask = np.zeros((256, 256), dtype=np.int64)
    mask[10:20, 100:105] = 1
    boxes = debris_boxes(mask)
    assert len(boxes) == 1
    assert boxes[0] == DebrisBox(xmin=100, ymin=10, xmax=104, ymax=19)


def test_two_separated_blobs_yield_two_boxes():
    mask = np.zeros((256, 256), dtype=np.int64)
    mask[5:8, 5:8] = 1
    mask[50:55, 200:210] = 1
    boxes = debris_boxes(mask)
    assert len(boxes) == 2


def test_min_area_filters_small_blobs():
    mask = np.zeros((256, 256), dtype=np.int64)
    mask[5:6, 5:6] = 1  # 1 px
    mask[20:22, 20:22] = 1  # 4 px
    assert len(debris_boxes(mask, min_area=3)) == 1


def test_debris_boxes_ignores_non_debris_classes():
    mask = np.full((256, 256), 7, dtype=np.int64)  # marine water only
    assert debris_boxes(mask) == []
    mask[100:102, 100:102] = 1
    assert len(debris_boxes(mask)) == 1


def test_to_yolo_lines_normalized_values():
    mask = np.zeros((256, 256), dtype=np.int64)
    mask[0:128, 0:128] = 1
    lines = to_yolo_lines(debris_boxes(mask))
    assert len(lines) == 1
    cls, cx, cy, w, h = (float(v) for v in lines[0].split())
    assert cls == 0.0
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
    assert 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0
    np.testing.assert_allclose([cx, cy, w, h], [0.25, 0.25, 0.5, 0.5], atol=1e-6)


def test_rgb_composite_shape_dtype():
    image = np.random.default_rng(1).random((11, 256, 256)).astype(np.float32)
    rgb = rgb_composite(image)
    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
    assert rgb.min() >= 0 and rgb.max() <= 255


def test_rgb_composite_sanitizes_nan():
    image = np.ones((11, 256, 256), dtype=np.float32)
    image[2, 10, 10] = np.nan
    rgb = rgb_composite(image)
    assert np.isfinite(rgb).all()