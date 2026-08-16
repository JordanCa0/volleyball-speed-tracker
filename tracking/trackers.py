"""Ball and player tracking.

Two different jobs, so two different mechanisms:

- the ball is a single object, and the problem is false positives, so the
  tracker's job is to pick which of this frame's candidates is the ball;
- the players are many, and the problem is identity, so the tracker's job
  is to keep a stable id on each one across frames.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker


class BallTracker:
    """Single-ball selection: which of this frame's candidates is the ball.

    Assumes exactly one ball is in play, which is what makes this a filter
    rather than a tracker — it never assigns ids and holds no notion of a
    track.

    Two strategies, because the original one does not survive contact with
    club footage:

    `"static"` (default) exploits the one thing every false positive here has
    in common — court lines, sponsor banners, scoreboards, benches and wall
    panels do not move. Any candidate that keeps reappearing within
    `static_radius` px of where it was over the last `static_window` frames is
    dropped, and the most confident survivor wins. Measured on 150 frames of
    club-gym footage this picks the ball on every frame where the ball was
    moving, against 2 of 5 for the centroid filter.

    `"centroid"` is the Roboflow ball-sports filter: buffer recent candidate
    positions and keep the one nearest their centroid. It works when the ball
    is usually the only candidate, and fails badly when it isn't — the buffer
    holds *every* candidate, so a cluster of static false positives drags the
    centroid onto itself and holds it there. Kept for comparison.

    The static strategy assumes a roughly fixed camera. Under a hard pan the
    court lines move too, so they stop looking static and start surviving the
    filter; that needs camera-motion compensation, which we do not do yet.
    """

    def __init__(self, buffer_size: int = 10, strategy: str = "static",
                 static_window: int = 25, static_radius: float = 18.0,
                 static_min_hits: int = 6, static_size_tol: float = 0.15):
        if strategy not in ("static", "centroid"):
            raise ValueError(f"unknown ball selection strategy: {strategy}")
        self.strategy = strategy
        self.buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self.history: deque[np.ndarray] = deque(maxlen=static_window)
        self.static_radius = static_radius
        self.static_min_hits = static_min_hits
        self.static_size_tol = static_size_tol

    def _static_mask(self, points: np.ndarray) -> np.ndarray:
        """True for candidates that keep turning up unchanged.

        "Unchanged" means both position *and* apparent size, and the size half
        is not optional. Filming from behind the server, a ball flying away
        holds almost the same image position for its whole flight — it shrinks
        rather than moves. Position-only staticness cannot tell that from a
        sponsor banner, and duly threw away 86% of the real detections on
        end-on footage. A banner holds its size; a receding ball does not.
        """
        if not self.history:
            return np.zeros(len(points), dtype=bool)
        past = np.concatenate(list(self.history))
        if len(past) == 0:
            return np.zeros(len(points), dtype=bool)

        close = (np.linalg.norm(past[None, :, :2] - points[:, None, :2], axis=2)
                 < self.static_radius)
        radii, past_radii = points[:, 2][:, None], past[:, 2][None, :]
        scale = np.maximum(np.maximum(radii, past_radii), 1e-6)
        same_size = np.abs(past_radii - radii) / scale <= self.static_size_tol
        return (close & same_size).sum(axis=1) >= self.static_min_hits

    @staticmethod
    def _points(detections: sv.Detections, xy: np.ndarray) -> np.ndarray:
        """(x, y, radius) per candidate — the state staticness is judged on."""
        widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        radii = np.maximum(widths, heights) / 2.0
        return np.column_stack([xy, radii])

    def _select_static(self, points: np.ndarray,
                       detections: sv.Detections) -> int | None:
        moving = ~self._static_mask(points)
        if not moving.any():
            # Everything on screen is furniture; better to report nothing than
            # to report a banner.
            return None
        confidence = (detections.confidence if detections.confidence is not None
                      else np.ones(len(points)))
        scores = np.where(moving, confidence, -1.0)
        return int(np.argmax(scores))

    def _select_centroid(self, xy: np.ndarray) -> int:
        centroid = np.mean(np.concatenate(list(self.buffer)), axis=0)
        return int(np.argmin(np.linalg.norm(xy - centroid, axis=1)))

    def update(self, detections: sv.Detections) -> sv.Detections:
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            self.history.append(np.zeros((0, 3)))
            return detections

        points = self._points(detections, xy)
        if self.strategy == "centroid":
            index = self._select_centroid(xy)
        else:
            index = self._select_static(points, detections)

        self.history.append(points)
        if index is None:
            return detections[np.zeros(len(detections), dtype=bool)]
        return detections[[index]]

    def reset(self) -> None:
        self.buffer.clear()
        self.history.clear()


class PlayerTracker:
    """Multi-object player tracking via ByteTrack.

    `frame_rate` matters: ByteTrack ages lost tracks in frames, so passing
    the video's real rate keeps `lost_track_buffer` meaning the same wall
    time across 30, 50 and 60fps footage.
    """

    def __init__(self, frame_rate: float = 30.0, lost_track_buffer: int = 30,
                 track_activation_threshold: float = 0.5,
                 minimum_consecutive_frames: int = 2,
                 minimum_iou_threshold: float = 0.1,
                 drop_unconfirmed: bool = True):
        self.tracker = ByteTrackTracker(
            frame_rate=frame_rate,
            lost_track_buffer=lost_track_buffer,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
        )
        self.drop_unconfirmed = drop_unconfirmed

    def update(self, detections: sv.Detections) -> sv.Detections:
        tracked = self.tracker.update(detections)
        if not self.drop_unconfirmed or tracked.tracker_id is None:
            return tracked
        # A track that hasn't yet been seen `minimum_consecutive_frames` times
        # is reported with id -1. Those are provisional and frequently spurious
        # (a fragment of crowd, a half-occluded bench player), so they are not
        # worth drawing or counting.
        return tracked[tracked.tracker_id != -1]

    def reset(self) -> None:
        self.tracker.reset()
