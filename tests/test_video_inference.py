"""Tests for src/detection/video_inference.py (drone video debris pipeline)."""
import json

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.detection.video_inference import (
    DroneDetector,
    FrameDetection,
    sample_frame_indices,
)


class _FakeDebrisModel(nn.Module):
    """Deterministic model: Marine Debris (channel 0) in bottom-right quadrant."""

    def forward(self, x):
        b, c, h, w = x.shape
        logits = torch.zeros(b, 11, h, w, device=x.device)
        logits[:, 6] = 1.0  # marine water
        logits[:, 0, h // 2:, w // 2:] = 5.0  # debris
        return logits


def _write_test_video(path, frames=12, fps=8.0, size=64):
    cap = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (size, size),
    )
    if not cap.isOpened():
        cap.release()
        path = path.with_name(path.stem + ".avi")
        cap = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (size, size),
        )
    for i in range(frames):
        frame = np.full((size, size, 3), int(30 + i * 3), dtype=np.uint8)
        cap.write(frame)
    cap.release()
    return path


def test_sample_frame_indices_honors_rate():
    idx = sample_frame_indices(total_frames=300, video_fps=30.0, sample_fps=15.0)
    assert len(idx) == 150
    assert idx[0] == 0 and idx[1] == 2
    assert idx[-1] <= 299


def test_sample_frame_indices_rejects_bad_inputs():
    assert sample_frame_indices(0, 30.0, 3.0) == []
    assert sample_frame_indices(100, 0.0, 3.0) == []
    assert sample_frame_indices(100, 30.0, 0.0) == []


def test_detect_frame_shapes_and_boxes(tmp_path):
    detector = DroneDetector(_FakeDebrisModel())
    frame = np.full((80, 120, 3), 0.5, dtype=np.float32)
    det = detector.detect_frame(frame)
    assert isinstance(det, FrameDetection)
    assert det.debris_mask.shape == (80, 120)
    assert det.concentration.shape == (80, 120)
    assert det.concentration.min() >= 0.0 and det.concentration.max() <= 1.0
    assert len(det.boxes) >= 1
    b = det.boxes[0]
    # bottom-right quadrant -> scaled coords should fall in lower-right half
    assert b.xmin >= 55 and b.ymin >= 35
    assert b.xmax <= 119 and b.ymax <= 79
    assert det.debris_pixels > 0
    assert 0.0 < det.debris_fraction < 0.5


def test_detect_frame_rejects_bad_shape():
    detector = DroneDetector(_FakeDebrisModel())
    with pytest.raises(ValueError):
        detector.detect_frame(np.zeros((10, 10, 4), dtype=np.uint8))


def test_overlay_returns_annotated_bgr(tmp_path):
    detector = DroneDetector(_FakeDebrisModel())
    frame = np.full((64, 64, 3), 100, dtype=np.uint8)
    det = detector.detect_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    out = detector.overlay(frame, det)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame)


def test_process_video_writes_output_and_report(tmp_path):
    video = _write_test_video(tmp_path / "drone.mp4", frames=12, fps=8.0)
    out = tmp_path / "drone_debris.mp4"
    report = tmp_path / "drone_report.json"
    detector = DroneDetector(_FakeDebrisModel())
    summary = detector.process_video(
        video, output_video=out, report=report, sample_fps=4.0
    )
    assert out.exists() and report.exists()
    assert summary.sampled_frames == 6  # 12 frames @ 8fps sampled @ 4fps
    assert summary.frames_with_debris == 6

    payload = json.loads(report.read_text())
    assert payload["summary"]["sampled_frames"] == 6
    assert len(payload["frames"]) == 6
    f0 = payload["frames"][0]
    assert f0["debris_pixels"] > 0
    assert f0["n_boxes"] >= 1
    assert summary.aggregate_concentration.shape == (64, 64)

    # verify output video can be opened and has the annotated frame dimensions
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    ret, frame = cap.read()
    cap.release()
    assert ret
    assert frame.shape[:2] == (64, 64)


def test_process_video_missing_input_raises(tmp_path):
    detector = DroneDetector(_FakeDebrisModel())
    with pytest.raises(IOError):
        detector.process_video(tmp_path / "missing.mp4")


def test_aggregate_concentration_mean(tmp_path):
    video = _write_test_video(tmp_path / "agg.mp4", frames=4, fps=4.0, size=48)
    detector = DroneDetector(_FakeDebrisModel())
    summary = detector.process_video(video, sample_fps=4.0)
    # all 4 frames identical -> aggregate equals a single frame's concentration
    cap = cv2.VideoCapture(str(video))
    ret, frame = cap.read()
    cap.release()
    single = detector.detect_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    np.testing.assert_allclose(
        summary.aggregate_concentration, single.concentration, atol=1e-3
    )


def test_detect_image_loads_drone_file_end_to_end(tmp_path):
    from src.data.drone_ingestion import read_rgb_image

    frame = np.full((96, 128, 3), 60, dtype=np.uint8)
    img_path = tmp_path / "drone.jpg"
    assert cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    detector = DroneDetector(_FakeDebrisModel())
    out_path = tmp_path / "drone_debris.png"
    det = detector.detect_image(img_path, output_path=out_path)

    loaded = read_rgb_image(img_path)
    assert loaded.shape == (3, 96, 128)  # loader yields (3, H, W) float32
    assert loaded.dtype == np.float32 and loaded.min() >= 0.0

    assert isinstance(det, FrameDetection)
    assert det.debris_mask.shape == (96, 128)
    assert det.concentration.shape == (96, 128)
    assert det.debris_pixels > 0
    assert det.debris_fraction > 0.0
    assert out_path.exists()


def test_sampled_frame_indices_are_readable_in_video(tmp_path):
    video = _write_test_video(tmp_path / "frames.mp4", frames=20, fps=10.0, size=32)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    indices = sample_frame_indices(total, video_fps, sample_fps=5.0)
    assert indices and indices[0] == 0
    assert indices == sorted(set(indices))  # no duplicates, ascending
    assert all(0 <= i < total for i in indices)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        assert ok, f"frame index {i} not readable"
        frames.append(frame)
    cap.release()
    assert len(frames) == len(indices)
    assert all(f.shape[:2] == (32, 32) for f in frames)