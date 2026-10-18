"""Drone/aerial video debris-detection inference pipeline (OpenCV).

Reads a video with ``cv2.VideoCapture``, samples frames at a configurable
rate (e.g. 2-5 fps), runs a trained segmentation model on each frame,
derives Marine Debris bounding boxes and a Gaussian concentration heatmap,
temporally aggregates the per-frame concentration, and writes an annotated
output video (boxes + heatmap overlay) plus a per-frame JSON report and an
aggregate pollution map PNG.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.data.drone_ingestion import to_model_channels
from src.mapping.heatmap import debris_concentration
from src.models.registry import resolve_variant

MODEL_SIZE = 256
DEBRIS_CHANNEL = 0
DEFAULT_SAMPLE_FPS = 3.0
_MAX_VIDEO_FRAMES = 3000  # guard against absurd total frame counts


@dataclass
class BoxRecord:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    area_px: int


@dataclass
class FrameDetection:
    debris_mask: np.ndarray  # float32 (H, W), full frame resolution, 0/1
    concentration: np.ndarray  # float32 (H, W) in [0, 1], full frame resolution
    boxes: list[BoxRecord]
    debris_pixels: int
    debris_fraction: float
    mean_concentration: float
    peak_concentration: float


@dataclass
class FrameReport:
    frame_index: int
    timestamp_s: float
    debris_pixels: int
    debris_fraction: float
    mean_concentration: float
    peak_concentration: float
    n_boxes: int
    boxes: list[dict] = field(default_factory=list)


@dataclass
class VideoSummary:
    input_path: Path
    output_path: Path | None
    report_path: Path
    total_frames: int
    sampled_frames: int
    sample_fps: float
    duration_s: float
    mean_concentration: float
    peak_concentration: float
    frames_with_debris: int
    debris_area_fraction_mean: float
    aggregate_concentration: np.ndarray  # mean over sampled frames, [0, 1]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("aggregate_concentration")
        d["input_path"] = str(d["input_path"])
        d["output_path"] = str(d["output_path"])
        d["report_path"] = str(d["report_path"])
        return d


def sample_frame_indices(total_frames: int, video_fps: float, sample_fps: float) -> list[int]:
    """Return frame indices to sample at ~``sample_fps`` from ``video_fps`` video."""
    if total_frames <= 0 or video_fps <= 0 or sample_fps <= 0:
        return []
    step = max(int(round(video_fps / max(sample_fps, 1e-6))), 1)
    return list(range(0, total_frames, step))


def _scale_boxes_to_frame(
    mask: np.ndarray, img_w: int, img_h: int, min_area: int = 1
) -> list[BoxRecord]:
    """Connected-component debris boxes from a (256, 256) mask, scaled to frame."""
    from scipy import ndimage

    debris = (mask > 0).astype(np.uint8)
    labeled, n = ndimage.label(debris)
    sx = img_w / mask.shape[1]
    sy = img_h / mask.shape[0]
    boxes: list[BoxRecord] = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labeled == i)
        area = int(ys.size)
        if area < min_area:
            continue
        xmin = int(round(xs.min() * sx))
        ymin = int(round(ys.min() * sy))
        xmax = int(round(xs.max() * sx))
        ymax = int(round(ys.max() * sy))
        xmax = min(xmax, img_w - 1)
        ymax = min(ymax, img_h - 1)
        boxes.append(BoxRecord(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, area_px=area))
    return boxes


class DroneDetector:
    """Runs a trained segmentation model on drone video frames.

    The wrapped model must accept ``(B, 11, H, W)`` and return ``(B, 11, H, W)``
    logits (channel 0 = Marine Debris), matching every registered variant.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        adapter_mode: str = "replicate",
        sigma: float = 2.0,
        min_box_area: int = 1,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.adapter_mode = adapter_mode
        self.sigma = sigma
        self.min_box_area = min_box_area

    @classmethod
    def from_checkpoint(
            cls,
            variant: str = "baseline",
            checkpoint: str | Path | None = None,
            device: torch.device | None = None,
            **kwargs,
        ) -> "VideoDebrisDetector":
        """Build a detector from a registered variant checkpoint."""
        if checkpoint is None:
            checkpoint = Path("artifacts") / variant / "best.pth"
        checkpoint = Path(checkpoint)
        model = resolve_variant(variant, pretrained=False)
        from src.utils.checkpoint import safe_load_state_dict
        safe_load_state_dict(model, checkpoint, map_location="cpu")
        detector = cls(model, device=device, **kwargs)
        detector.checkpoint_epoch = None
        return detector

    @torch.no_grad()
    def detect_frame(self, rgb_hwc: np.ndarray) -> FrameDetection:
        """Detect debris on a single RGB frame ``(H, W, 3)`` uint8 or float32.

        The frame is resized to ``MODEL_SIZE`` for inference; the mask and
        concentration are resized back to the original resolution.
        """
        frame = np.asarray(rgb_hwc)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB frame, got {frame.shape}")
        img_h, img_w = frame.shape[:2]

        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)

        small = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
        small_chw = small.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = to_model_channels(small_chw, mode=self.adapter_mode)
        batch = (
            torch.from_numpy(np.ascontiguousarray(input_tensor[None])).to(self.device, non_blocking=True)
        )
        with torch.autocast(device_type=self.device.type, enabled=True):
            logits = self.model(batch)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)
        debris_mask_256 = (pred == DEBRIS_CHANNEL).astype(np.float32)

        concentration_256 = debris_concentration(debris_mask_256, sigma=self.sigma)
        debris_mask = cv2.resize(
            debris_mask_256, (img_w, img_h), interpolation=cv2.INTER_NEAREST
        )
        concentration = cv2.resize(
            concentration_256, (img_w, img_h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        boxes = _scale_boxes_to_frame(debris_mask_256, img_w, img_h, self.min_box_area)
        debris_pixels = int(debris_mask.sum())
        total_px = max(img_h * img_w, 1)
        return FrameDetection(
            debris_mask=debris_mask,
            concentration=concentration,
            boxes=boxes,
            debris_pixels=debris_pixels,
            debris_fraction=debris_pixels / total_px,
            mean_concentration=float(concentration.mean()),
            peak_concentration=float(concentration.max()),
        )

    def overlay(
        self,
        frame_bgr: np.ndarray,
        detection: FrameDetection,
        draw_boxes: bool = True,
        heatmap: bool = True,
        heatmap_alpha: float = 0.45,
        heatmap_threshold: float = 0.05,
    ) -> np.ndarray:
        """Annotate a BGR frame with a heatmap overlay and debris boxes."""
        out = frame_bgr.copy()
        if heatmap and detection.concentration.max() > 0:
            vis = (np.clip(detection.concentration, 0.0, 1.0) * 255.0).astype(np.uint8)
            heat = cv2.applyColorMap(vis, cv2.COLORMAP_HOT)
            mask = detection.concentration > heatmap_threshold
            if mask.any():
                out[mask] = (
                    out[mask].astype(np.float32) * (1.0 - heatmap_alpha)
                    + heat[mask].astype(np.float32) * heatmap_alpha
                ).astype(np.uint8)
        if draw_boxes:
            for b in detection.boxes:
                cv2.rectangle(out, (b.xmin, b.ymin), (b.xmax, b.ymax), (0, 0, 255), 2)
                label = "debris"
                cv2.putText(
                    out, label, (b.xmin, max(b.ymin - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                )
        return out

    def detect_image(self, image_path: Path, output_path: Path | None = None) -> FrameDetection:
        """Run detection on a single drone/aerial image file.

        Optionally writes the annotated PNG and a heatmap-only PNG.
        """
        from src.data.drone_ingestion import read_rgb_image

        rgb_chw = read_rgb_image(image_path)
        rgb_hwc = np.ascontiguousarray(rgb_chw.transpose(1, 2, 0)) * 255.0
        detection = self.detect_frame(rgb_hwc.astype(np.uint8))

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            annotated = self.overlay(rgb_hwc.astype(np.uint8), detection)
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        return detection

    def process_video(
        self,
        input_video: Path,
        output_video: Path | None = None,
        report: Path | None = None,
        sample_fps: float = DEFAULT_SAMPLE_FPS,
        draw_boxes: bool = True,
        heatmap: bool = True,
        aggregate_png: Path | None = None,
    ) -> VideoSummary:
        """Process a video end-to-end and write annotated output + report.

        If ``output_video`` is None a ``_debris.mp4`` file is written next to
        the input; if ``report`` is None a ``_debris_report.json`` file is
        written next to the input. ``aggregate_png`` defaults to the report
        path with a .png extension.
        """
        input_video = Path(input_video)
        if output_video is None:
            output_video = input_video.with_name(input_video.stem + "_debris.mp4")
        if report is None:
            report = input_video.with_name(input_video.stem + "_debris_report.json")
        if aggregate_png is None:
            aggregate_png = report.with_suffix(".png")
        output_video = Path(output_video)
        report = Path(report)
        aggregate_png = Path(aggregate_png)

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise IOError(f"could not open video: {input_video}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = min(total_frames, _MAX_VIDEO_FRAMES)
        indices = sample_frame_indices(total_frames, video_fps, sample_fps)
        if not indices:
            cap.release()
            raise ValueError(f"video has no frames: {input_video}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first = cap.read()
        if not ret:
            cap.release()
            raise IOError(f"could not read first frame: {input_video}")
        frame_h, frame_w = first.shape[:2]

        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            sample_fps,
            (frame_w, frame_h),
        )
        if not writer.isOpened():
            writer.release()
            raise IOError(f"could not open video writer for {output_video}")

        frame_reports: list[FrameReport] = []
        agg = np.zeros((frame_h, frame_w), dtype=np.float32)
        frames_with_debris = 0
        with torch.no_grad():
            for camera_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, camera_idx)
                ret, frame_bgr = cap.read()
                if not ret:
                    continue
                rgb_hwc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detection = self.detect_frame(rgb_hwc)
                annotated = self.overlay(frame_bgr, detection, draw_boxes=draw_boxes, heatmap=heatmap)
                writer.write(annotated)

                agg += detection.concentration
                if detection.debris_pixels > 0:
                    frames_with_debris += 1
                frame_reports.append(
                    FrameReport(
                        frame_index=int(camera_idx),
                        timestamp_s=round(camera_idx / video_fps, 3),
                        debris_pixels=detection.debris_pixels,
                        debris_fraction=round(detection.debris_fraction, 6),
                        mean_concentration=round(detection.mean_concentration, 6),
                        peak_concentration=round(detection.peak_concentration, 6),
                        n_boxes=len(detection.boxes),
                        boxes=[asdict(b) for b in detection.boxes],
                    )
                )
        writer.release()
        cap.release()

        n = max(len(frame_reports), 1)
        aggregate_concentration = (agg / np.float32(n)).astype(np.float32)
        summary = VideoSummary(
            input_path=input_video,
            output_path=output_video,
            report_path=report,
            total_frames=total_frames,
            sampled_frames=len(frame_reports),
            sample_fps=sample_fps,
            duration_s=round(total_frames / video_fps, 3),
            mean_concentration=round(float(aggregate_concentration.mean()), 6),
            peak_concentration=round(float(aggregate_concentration.max()), 6),
            frames_with_debris=frames_with_debris,
            debris_area_fraction_mean=round(
                float(np.mean([f.debris_fraction for f in frame_reports])), 6
            ),
            aggregate_concentration=aggregate_concentration,
        )

        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {"summary": summary.to_dict(), "frames": [asdict(f) for f in frame_reports]},
                indent=2,
            )
        )

        if aggregate_png is not None:
            aggregate_png.parent.mkdir(parents=True, exist_ok=True)
            vis = (np.clip(aggregate_concentration, 0.0, 1.0) * 255.0).astype(np.uint8)
            cv2.imwrite(
                str(aggregate_png),
                cv2.applyColorMap(vis, cv2.COLORMAP_HOT),
            )

        print(
            f"video {input_video.name}: sampled {summary.sampled_frames} frames, "
            f"{frames_with_debris} with debris, mean concentration "
            f"{summary.mean_concentration}, peak {summary.peak_concentration}"
        )
        return summary