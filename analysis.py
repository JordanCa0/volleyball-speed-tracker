"""End-to-end analysis of a clip: video in, ball speeds out.

One entry point shared by the CLI (track.py) and the web UI (webapp.py), so
both measure identically — a speed that depends on which front end asked for
it would be worse than no speed at all.

    frame -> sliced detection -> shape gate -> static suppression
          -> depth from apparent size -> 3D trajectory fit -> peak speed
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from calibration import CameraIntrinsics, focal_length_from_fov
from speed import BallSample, Hit, SpeedEngine
from tracking.annotators import BallAnnotator
from tracking.detectors import SPORTS_BALL_CLASS_ID, BallDetector, YoloBackend
from tracking.trackers import BallTracker

DEFAULT_BALL_MODEL = "models/volleyball_ball.pt"

# How close to the edge counts as "leaving frame". A ball whose box touches
# this margin is likely to be half out already, and its apparent radius is then
# clipped — which would read as a sudden jump in depth.
EDGE_MARGIN_PX = 24


@dataclass
class AnalysisResult:
    source: str
    width: int
    height: int
    fps: float
    frames_processed: int
    frames_with_ball: int
    hits: list[Hit] = field(default_factory=list)
    elapsed_s: float = 0.0
    annotated_path: str | None = None
    intrinsics_source: str = "reference"

    @property
    def reliable_hits(self) -> list[Hit]:
        return [h for h in self.hits if h.reliable]

    @property
    def fastest(self) -> Hit | None:
        candidates = self.reliable_hits or self.hits
        return max(candidates, key=lambda h: h.peak_speed_kmh) if candidates else None

    def summary(self) -> str:
        lines = [
            f"{self.frames_processed} frames at {self.fps:.1f}fps "
            f"({self.elapsed_s:.0f}s, {self.frames_processed/max(self.elapsed_s,1e-9):.1f} fps)",
            f"ball located in {self.frames_with_ball}/{self.frames_processed} frames "
            f"({100*self.frames_with_ball/max(self.frames_processed,1):.0f}%)",
            f"{len(self.hits)} flight segments, {len(self.reliable_hits)} reliable",
        ]
        if self.fastest:
            lines.append(f"fastest: {self.fastest.peak_speed_kmh:.1f} km/h")
        return "\n".join(lines)


def ball_radius_px(xyxy: np.ndarray) -> float:
    """Radius from the bounding box.

    Measured against mask area and mask minor axis on real end-on footage,
    the box is by far the most stable: 3.2% frame-to-frame noise against 11.7%
    and 20.3%. Since depth error equals radius error, that difference is the
    difference between a 2.7% and a 10% speed error, so the box wins despite
    the mask looking like the more sophisticated choice.
    """
    x1, y1, x2, y2 = xyxy
    return max(x2 - x1, y2 - y1) / 2.0


def touches_edge(xyxy: np.ndarray, width: int, height: int,
                 margin: int = EDGE_MARGIN_PX) -> bool:
    x1, y1, x2, y2 = xyxy
    return bool(x1 <= margin or y1 <= margin
                or x2 >= width - margin or y2 >= height - margin)


def default_intrinsics(width: int, height: int) -> CameraIntrinsics:
    """Fallback when the user has not run the ruler calibration.

    A 70° horizontal field of view is typical of a phone main camera. Speeds
    computed from it scale linearly with the true focal length, so they are
    indicative, not measured — the UI says so.
    """
    return focal_length_from_fov((width, height), 70.0)


def analyse_video(source: str,
                  intrinsics: CameraIntrinsics | None = None,
                  ball_model: str = DEFAULT_BALL_MODEL,
                  confidence: float = 0.25,
                  imgsz: int = 1024,
                  max_side_px: float | None = 60.0,
                  max_aspect: float | None = 1.6,
                  slice_inference: bool = True,
                  max_frames: int | None = None,
                  start_frame: int = 0,
                  device: str | None = None,
                  annotate_path: str | None = None,
                  progress: Callable[[int, int | None], None] | None = None,
                  ) -> AnalysisResult:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if total:
            total = max(total - start_frame, 0)
    if max_frames is not None:
        total = min(total, max_frames) if total else max_frames

    intrinsics_source = "reference"
    if intrinsics is None:
        intrinsics = default_intrinsics(width, height)
        intrinsics_source = "assumed 70° FOV"

    backend = YoloBackend(ball_model, SPORTS_BALL_CLASS_ID, confidence, imgsz, device)
    detector = BallDetector(backend, frame_wh=(width, height),
                            slice_inference=slice_inference,
                            max_side_px=max_side_px, max_aspect=max_aspect)
    tracker = BallTracker()
    engine = SpeedEngine(intrinsics)
    annotator = BallAnnotator(radius=12, buffer_size=5) if annotate_path else None

    writer = None
    if annotate_path:
        writer = cv2.VideoWriter(annotate_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))

    frames = 0
    frames_with_ball = 0
    started = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = tracker.update(detector(frame))
            # File mode: derive time from the frame index, which is exact by
            # definition and keeps replays deterministic (TDD §6.6).
            timestamp = (start_frame + frames) / fps

            if len(detections):
                frames_with_ball += 1
                xyxy = detections.xyxy[0]
                x1, y1, x2, y2 = xyxy
                sample = BallSample(
                    frame_idx=start_frame + frames,
                    timestamp=timestamp,
                    center_px=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    radius_px=ball_radius_px(xyxy),
                    confidence=float(detections.confidence[0])
                    if detections.confidence is not None else 1.0,
                )
                engine.update(sample, left_frame=touches_edge(xyxy, width, height))
            else:
                engine.update(None)

            if writer is not None:
                writer.write(annotator.annotate(frame, detections))

            frames += 1
            if progress is not None and frames % 10 == 0:
                progress(frames, total)
            if max_frames and frames >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    engine.finish()
    return AnalysisResult(
        source=source, width=width, height=height, fps=fps,
        frames_processed=frames, frames_with_ball=frames_with_ball,
        hits=engine.hits, elapsed_s=time.monotonic() - started,
        annotated_path=annotate_path, intrinsics_source=intrinsics_source,
    )


def hits_to_csv(hits: list[Hit]) -> str:
    rows = ["start_s,end_s,peak_kmh,peak_mph,points,mean_depth_m,reliable"]
    for hit in hits:
        rows.append(
            f"{hit.start_time:.3f},{hit.end_time:.3f},"
            f"{hit.peak_speed_kmh:.1f},{hit.peak_speed_kmh*0.621371:.1f},"
            f"{hit.n_points},{hit.mean_depth_m:.2f},{int(hit.reliable)}"
        )
    return "\n".join(rows)
