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
    """Single-ball false-positive filter (Roboflow ball-sports approach).

    Buffers the candidate positions of recent frames and keeps, each frame,
    only the detection nearest the centroid of that buffer. Assumes exactly
    one ball is in play, which is what makes it a filter rather than a
    tracker — it never assigns ids and holds no notion of a track.

    The buffer is a lag: the centroid sits behind a moving ball by roughly
    half the buffer's duration of travel. That is harmless while the ball is
    the only candidate (`argmin` returns it regardless) and costly when it
    isn't, so `buffer_size` trades false-positive rejection against how fast
    a ball the filter can follow.
    """

    def __init__(self, buffer_size: int = 10):
        self.buffer: deque[np.ndarray] = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = int(np.argmin(distances))
        return detections[[index]]

    def reset(self) -> None:
        self.buffer.clear()


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
