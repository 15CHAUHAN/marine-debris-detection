"""Tests for the TACO YOLOv8 RGB debris routing in dashboard/pipeline.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import pipeline


class _FakeBoxes:
    def __init__(self, rows):
        self.xyxy = [r[:4] for r in rows]
        self.conf = [r[4] for r in rows]
        self.cls = [r[5] for r in rows]

    def __len__(self):
        return len(self.xyxy)


class _FakeResult:
    def __init__(self, rows):
        self.boxes = _FakeBoxes(rows)


class _FakeYolo:
    """Minimal stand-in for an ultralytics YOLO model."""

    def __init__(self, rows):
        self._rows = rows

    def predict(self, rgb, **kwargs):
        return [_FakeResult(self._rows)]


def test_taco_classes_are_the_seven_macro_classes():
    assert pipeline.TACO_CLASSES == [
        "Bottle", "Can", "Plastic bag", "Cup",
        "Wrapper_Sachet", "Container", "Other_Trash",
    ]
    assert set(pipeline.TACO_PRIMARY_CLASSES) <= set(pipeline.TACO_CLASSES)


def test_boxes_to_debris_mask_paints_and_clips():
    boxes = [{"xmin": -5, "ymin": -5, "xmax": 4, "ymax": 4}]
    mask = pipeline.boxes_to_debris_mask(boxes, 10, 10)
    assert mask.shape == (10, 10)
    assert mask[0:5, 0:5].all()  # clipped to the top-left corner
    assert mask[5:, :].sum() == 0.0


def test_boxes_to_debris_mask_ignores_degenerate_boxes():
    boxes = [{"xmin": 5, "ymin": 5, "xmax": 5, "ymax": 5}]
    assert pipeline.boxes_to_debris_mask(boxes, 10, 10).sum() == 0.0


def test_taco_boxes_to_detection_maps_names_sorts_and_concentrates():
    detections = [
        (2, 2, 10, 8, 0.90, 0),    # Bottle, 9x7 = 63 px
        (20, 20, 40, 40, 0.55, 1),  # Can, 21x21 = 441 px
        (30, 30, 33, 33, 0.30, 2),  # Plastic bag, 4x4 = 16 px (inside Can)
    ]
    det = pipeline.taco_boxes_to_detection(detections, 64, 64, label="beach.jpg")
    assert det["source"] == "taco"
    assert det["label"] == "beach.jpg"
    assert [b["class_name"] for b in det["boxes"]] == ["Can", "Bottle", "Plastic bag"]
    assert det["class_counts"] == {"Can": 1, "Bottle": 1, "Plastic bag": 1}
    assert det["boxes"][0]["area_px"] == 441
    # The Plastic bag box sits inside the Can box, so the union mask is 63 + 441.
    assert det["debris"].sum() == 63 + 441
    assert det["concentration"].shape == (64, 64)
    assert det["stats"]["severity"] in ("Moderate", "High", "Critical", "Low")


def test_taco_boxes_to_detection_empty_is_clean_water():
    det = pipeline.taco_boxes_to_detection([], 32, 32)
    assert det["boxes"] == []
    assert det["debris"].sum() == 0.0
    assert det["concentration"].sum() == 0.0
    assert det["stats"]["density_pct"] == 0.0
    assert det["stats"]["severity"] == "Low"
    assert det["rgb"].shape == (32, 32, 3)


def test_taco_boxes_to_detection_unknown_class_falls_back_to_other_trash():
    det = pipeline.taco_boxes_to_detection([(1, 1, 4, 4, 0.8, 99)], 16, 16)
    assert det["boxes"][0]["class_name"] == "Other_Trash"


def test_taco_detector_available_false_without_checkpoint(tmp_path):
    assert pipeline.taco_detector_available(tmp_path / "missing.pt") is False


def test_taco_detector_available_true_with_checkpoint(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"stub")
    assert pipeline.taco_detector_available(ckpt) is True


def test_detect_upload_parses_ultralytics_boxes_without_torch_model():
    detector = pipeline.TacoYoloDetector(model_path="does-not-exist.pt")
    detector._model = _FakeYolo([(10, 10, 50, 50, 0.91, 0), (60, 60, 90, 90, 0.42, 1)])
    import cv2

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    det = detector.detect_upload(buf.tobytes(), filename="drone.png")
    assert det["label"] == "drone.png"
    assert {b["class_name"] for b in det["boxes"]} == {"Bottle", "Can"}
    assert det["boxes"][0]["area_px"] == 41 * 41


def test_detect_image_accepts_float_rgb_and_returns_uint8():
    detector = pipeline.TacoYoloDetector(model_path="does-not-exist.pt")
    detector._model = _FakeYolo([(1, 1, 8, 8, 0.7, 2)])
    det = detector.detect_image(np.full((16, 16, 3), 0.5, dtype=np.float32))
    assert det["rgb"].dtype == np.uint8
    assert det["boxes"][0]["class_name"] == "Plastic bag"


def test_detect_image_rejects_non_rgb():
    detector = pipeline.TacoYoloDetector(model_path="does-not-exist.pt")
    detector._model = _FakeYolo([])
    with pytest.raises(ValueError):
        detector.detect_image(np.zeros((16, 16), dtype=np.uint8))


def test_load_taco_detector_without_checkpoint_reports_unavailable(tmp_path):
    detector = pipeline.load_taco_detector(model_path=tmp_path / "best.pt")
    assert detector.available is False
    with pytest.raises(FileNotFoundError):
        detector.detect_upload(b"not-an-image")


class _RoutingStub:
    def __init__(self):
        self.calls = []

    def detect_upload(self, source_bytes, filename="", **kwargs):
        self.calls.append(("upload", filename))
        return {"source": "taco", "kind": "image"}

    def detect_video(self, source_bytes, source_name="", sample_fps=2.0, max_frames=60, **kwargs):
        self.calls.append(("video", source_name, sample_fps, max_frames))
        return {"source": "taco", "kind": "video"}


def test_analyze_with_taco_routes_images_and_videos():
    stub = _RoutingStub()
    img = pipeline.analyze_with_taco("drone", b"bytes", filename="a.jpg", detector=stub)
    assert img["kind"] == "image"
    assert stub.calls[-1] == ("upload", "a.jpg")

    vid = pipeline.analyze_with_taco("video", b"bytes", filename="b.mp4", detector=stub)
    assert vid["kind"] == "video"
    assert stub.calls[-1][0] == "video"


def test_analyze_with_taco_rejects_unsupported_source():
    with pytest.raises(ValueError):
        pipeline.analyze_with_taco("satellite", b"bytes", detector=_RoutingStub())


def test_draw_taco_boxes_writes_class_label():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = [{"xmin": 5, "ymin": 5, "xmax": 30, "ymax": 30, "confidence": 0.87,
              "class_name": "Bottle"}]
    out = pipeline.draw_taco_boxes(rgb, boxes)
    assert out.dtype == np.uint8
    assert (out[5:31, 5:6] > 0).any()
    assert out.sum() > 0  # label background/text drawn


class _FakeRgbBox:
    def __init__(self, xyxy, conf, cls):
        import torch

        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.conf = torch.tensor([conf], dtype=torch.float32)
        self.cls = torch.tensor([cls], dtype=torch.int64)


class _FakeRgbResult:
    def __init__(self):
        self.boxes = [_FakeRgbBox([1, 2, 10, 20], 0.9, 0)]
        self.names = {0: "Bottle"}

    def plot(self):
        # Distinctive BGR layout: channel 0 == 10, channel 2 == 20.
        return np.dstack([
            np.full((30, 30), 10, np.uint8),
            np.zeros((30, 30), np.uint8),
            np.full((30, 30), 20, np.uint8),
        ])


class _FakeRgbYolo:
    def __call__(self, image, conf=0.4, verbose=False):
        return [_FakeRgbResult()]


def test_predict_taco_rgb_returns_rgb_and_detections(monkeypatch):
    monkeypatch.setattr(pipeline, "_load_taco_rgb_model", lambda path: _FakeRgbYolo())
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    annotated, boxes = pipeline.predict_taco_rgb(image)
    # BGR -> RGB swap: the original channel 2 (value 20) must land in channel 0.
    assert annotated[0, 0, 0] == 20 and annotated[0, 0, 2] == 10
    assert boxes == [{
        "class": "Bottle", "confidence": 0.9,
        "xmin": 1, "ymin": 2, "xmax": 10, "ymax": 20,
    }]


def test_load_taco_rgb_model_caches_by_path(monkeypatch, tmp_path):
    import ultralytics

    created = []

    class _CountingYolo:
        def __init__(self, path):
            created.append(path)

    monkeypatch.setattr(ultralytics, "YOLO", _CountingYolo)
    pipeline._TACO_RGB_MODEL_CACHE.clear()
    weights = tmp_path / "best.pt"
    first = pipeline._load_taco_rgb_model(weights)
    second = pipeline._load_taco_rgb_model(weights)
    assert first is second
    assert created == [str(weights)]
    pipeline._TACO_RGB_MODEL_CACHE.clear()
