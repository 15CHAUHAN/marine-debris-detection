"""Reassemble drone/coastal demo videos from the UTD2 frame dataset.

The UTD2 Roboflow export (``utd2.v8i.yolo26.zip``) contains only annotated
JPEG frames, not the source drone videos. This script reconstructs a demo
mp4 from a set of consecutive frames belonging to one source clip (filenames
like ``train/images/vid_000284_frame0000028_jpg.rf.<hash>.jpg``), so the
footage can be uploaded to the dashboard's video-input tab.

Usage:
  python scripts/make_demo_video.py --list
  python scripts/make_demo_video.py --source vid_000284 --fps 25 --max-frames 200
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = ROOT / "utd2.v8i.yolo26.zip"
DEFAULT_OUT = ROOT / "artifacts" / "demo"

_SOURCE_RE = re.compile(r"^(?P<src>[A-Za-z0-9_]+)_frame\d+")
_FRAME_PNG_RE = re.compile(r"^frame_(?P<idx>\d+)_png$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _entry_base(name: str) -> str:
    """Return the roboflow base filename (everything before the .rf.<hash>)."""
    return name.split(".rf.")[0]


def _normalized(name: str) -> str:
    """Strip the .rf.<hash> suffix and trailing roboflow extension suffix.

    ``frame_00795_png_jpg.rf.<hash>.jpg`` -> ``frame_00795_png``.
    """
    base = Path(_entry_base(name)).name
    return re.sub(r"_(jpg|jpeg|png)$", "", base)


def _source_of(base: str) -> str:
    m = _SOURCE_RE.match(base)
    if m:
        return m.group("src")
    m = _FRAME_PNG_RE.match(base)
    if m:
        return "frame_png_clip"
    return ""


def _frame_index(norm: str) -> int:
    """Extract the frame number from a normalized filename."""
    digits = re.findall(r"\d+", norm)
    return int(digits[-1]) if digits else 0


def group_sources(entries: list[str]) -> dict[str, dict]:
    """Group image-relative entries by source clip with sorted frame indices.

    Returns ``{source_id: {"frames": [int, ...], "count": int}}``. Duplicate
    frame indices within a source are collapsed to the first occurrence.
    """
    groups: dict[str, dict] = {}
    for name in entries:
        if not Path(name).suffix.lower() in _IMAGE_EXTS or "/images/" not in name:
            continue
        norm = _normalized(name)
        base = _source_of(norm)
        if not base:
            continue
        idx = _frame_index(norm)
        g = groups.setdefault(base, {"frames": [], "count": 0})
        if idx not in g["frames"]:
            g["frames"].append(idx)
            g["count"] += 1
    for g in groups.values():
        g["frames"].sort()
    return groups


def list_sources(zip_path: Path) -> dict[str, dict]:
    """Return grouped sources for the given UTD2 zip archive."""
    with zipfile.ZipFile(zip_path) as zf:
        return group_sources(zf.namelist())


def write_source_video(
    zip_path: Path,
    source: str,
    out_dir: Path,
    fps: float = 25.0,
    max_frames: int = 0,
) -> Path:
    """Write an mp4 from a source clip's frames in frame order.

    ``max_frames=0`` means all frames of the clip. Returns the output path.
    """
    with zipfile.ZipFile(zip_path) as zf:
        groups = group_sources(zf.namelist())
        if source not in groups:
            raise KeyError(f"source {source!r} not found in {zip_path}")
        indices = groups[source]["frames"]
        if max_frames and max_frames > 0:
            indices = indices[:max_frames]
        if not indices:
            raise ValueError(f"source {source!r} has no frames")
        idx_map: dict[int, str] = {}
        for name in zf.namelist():
            if "/images/" not in name or not Path(name).suffix.lower() in _IMAGE_EXTS:
                continue
            norm = _normalized(name)
            if _source_of(norm) != source:
                continue
            idx = _frame_index(norm)
            idx_map.setdefault(idx, name)
        missing = [i for i in indices if i not in idx_map]
        if missing:
            raise ValueError(f"source {source!r} missing frames: {missing[:5]}...")

        writer: cv2.VideoWriter | None = None
        written = 0
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{source}.mp4"
        for idx in indices:
            data = np.frombuffer(zf.read(idx_map[idx]), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"could not decode frame {idx} ({idx_map[idx]})")
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (w, h),
                )
                if not writer.isOpened():
                    writer.release()
                    raise IOError(f"could not open video writer for {out}")
            elif frame.shape[:2] != (h, w):
                writer.release()
                raise ValueError(f"inconsistent frame size at index {idx}")
            writer.write(frame)
            written += 1
        if writer is not None:
            writer.release()
        print(
            f"wrote {out} ({written} frames, {fps:g} fps, source {source!r})"
        )
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="path to utd2.v8i.yolo26.zip")
    parser.add_argument("--list", action="store_true", help="list source clips and their frame counts")
    parser.add_argument("--source", default="", help="source clip id (default: clip with most frames)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for the mp4")
    parser.add_argument("--fps", type=float, default=25.0, help="mp4 frame rate")
    parser.add_argument("--max-frames", type=int, default=0, help="cap frames per clip (0 = all)")
    args = parser.parse_args()

    if not args.zip.exists():
        parser.error(f"zip not found: {args.zip} (pass --zip)")

    groups = list_sources(args.zip)
    if not groups:
        parser.error(f"no drone/coastal frames found in {args.zip}")

    if args.list:
        print(f"{len(groups)} source clips in {args.zip.name}:")
        for src in sorted(groups, key=lambda s: -groups[s]["count"]):
            g = groups[src]
            span = f"[{g['frames'][0]}..{g['frames'][-1]}]"
            print(f"  {src:<22} {g['count']:>5} frames  {span}")
        return

    source = args.source or max(groups, key=lambda s: groups[s]["count"])
    write_source_video(args.zip, source, args.out, fps=args.fps, max_frames=args.max_frames)
    print(f"Upload {args.out / (source + '.mp4')} to the dashboard Video tab to test detection.")


if __name__ == "__main__":
    main()