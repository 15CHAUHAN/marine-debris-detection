"""Tests for scripts/make_demo_video.py (UTD2 frames -> demo mp4)."""
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.make_demo_video import group_sources, write_source_video  # noqa: E402


def _entries():
    return [
        "train/images/vid_000284_frame0000003_jpg.rf.111.jpg",
        "train/images/vid_000284_frame0000001_jpg.rf.222.jpg",
        "train/images/vid_000284_frame0000002_jpg.rf.333.jpg",
        "train/images/vid_000284_frame0000002_jpg.rf.444.jpg",
        "valid/images/bio0015_frame0000142_jpg.rf.555.jpg",
        "test/images/obj1656_frame0000001_jpg.rf.666.jpg",
        "train/images/frame_00795_png_jpg.rf.777.jpg",
        "train/images/frame_00800_png_jpg.rf.888.jpg",
        "train/labels/vid_000284_frame0000001_jpg.rf.222.txt",
        "train/images/plastic21_jpg.rf.999.jpg",
        "train/images/11_png_jpg.rf.123.jpg",
    ]


def test_group_sources_groups_and_sorts_frames():
    groups = group_sources(_entries())
    assert groups["vid_000284"]["count"] == 3
    assert groups["vid_000284"]["frames"] == [1, 2, 3]
    assert groups["bio0015"]["frames"] == [142]
    assert groups["obj1656"]["frames"] == [1]
    assert groups["frame_png_clip"]["frames"] == [795, 800]
    assert "plastic21" not in groups
    assert "11_png" not in groups


def test_group_sources_ignores_labels_and_non_frames():
    groups = group_sources(_entries())
    for src, g in groups.items():
        for idx in g["frames"]:
            assert idx >= 0
    total = sum(g["count"] for g in groups.values())
    assert total == 7


def _synthetic_zip(path: Path, frames: int = 6, prefix: str = "vid_999999"):
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(frames):
            frame = np.full((48, 64, 3), int(40 + i * 10), dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", frame)
            assert ok
            zf.writestr(
                f"train/images/{prefix}_frame{i:07d}_jpg.rf.abc{i}.jpg", buf.tobytes()
            )


def test_write_source_video_builds_readable_mp4(tmp_path):
    zpath = tmp_path / "utd2.zip"
    _synthetic_zip(zpath, frames=6)
    out_dir = tmp_path / "out"
    out = write_source_video(zpath, "vid_999999", out_dir, fps=10.0)
    assert out.exists()
    assert out.suffix == ".mp4"

    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FPS) == 10.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n == 6
    ret, frame = cap.read()
    cap.release()
    assert ret
    assert frame.shape == (48, 64, 3)


def test_write_source_video_max_frames_cap(tmp_path):
    zpath = tmp_path / "utd2.zip"
    _synthetic_zip(zpath, frames=6)
    out = write_source_video(zpath, "vid_999999", tmp_path / "out2", fps=10.0, max_frames=2)
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == 2


def test_write_source_video_unknown_source_raises(tmp_path):
    zpath = tmp_path / "utd2.zip"
    _synthetic_zip(zpath, frames=2)
    with pytest.raises(KeyError):
        write_source_video(zpath, "vid_000000", tmp_path / "out3")