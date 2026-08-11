"""
Frame-to-frame ball tracking (TDD §6.2).

Strings per-frame detections into a gap-tolerant track. Gating is by
maximum plausible ball speed, not predicted position — a hit is a velocity
discontinuity that position prediction would wrongly reject.
"""
from collections import deque
from dataclasses import dataclass


@dataclass
class Detection:
    frame_idx: int
    timestamp: float                 # seconds
    center_px: tuple[float, float]
    radius_px: float


@dataclass
class TrackerResult:
    accepted: Detection | None       # detection admitted to the track this frame
    track_id: int
    track_ended: bool                # this update closed the current track


class Tracker:
    def __init__(self, max_speed_px_per_s: float, max_gap_frames: int = 5,
                 slack_px: float = 20.0, trail_len: int = 60):
        self.max_speed_px_per_s = max_speed_px_per_s
        self.max_gap_frames = max_gap_frames
        self.slack_px = slack_px
        self.trail: deque[tuple[float, float]] = deque(maxlen=trail_len)
        self._last: Detection | None = None
        self._gap = 0
        self._track_id = 0

    def update_multi(self, candidates: list[Detection]) -> TrackerResult:
        """Pick the best candidate given the track so far, then update.

        With an active track, continuity wins: the closest plausible
        candidate to where the ball already is. Without one, fall back to
        the first candidate (callers pass them ball-likeness-ordered).
        """
        if not candidates:
            return self.update(None)
        if self._last is None:
            return self.update(candidates[0])
        lx, ly = self._last.center_px
        reachable = []
        for c in candidates:
            dt = c.timestamp - self._last.timestamp
            if dt <= 0:
                continue
            dist = ((c.center_px[0] - lx) ** 2 + (c.center_px[1] - ly) ** 2) ** 0.5
            if dist <= self.max_speed_px_per_s * dt + self.slack_px:
                reachable.append((dist, c))
        if not reachable:
            return self.update(None)
        return self.update(min(reachable, key=lambda p: p[0])[1])

    def update(self, detection: Detection | None) -> TrackerResult:
        if detection is not None:
            if self._last is None:
                return self._accept(detection)
            dt = detection.timestamp - self._last.timestamp
            if dt > 0:
                dx = detection.center_px[0] - self._last.center_px[0]
                dy = detection.center_px[1] - self._last.center_px[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= self.max_speed_px_per_s * dt + self.slack_px:
                    return self._accept(detection)
            # Implausible jump (or non-monotonic timestamp): treat as a miss.
        return self._miss()

    def _accept(self, detection: Detection) -> TrackerResult:
        self._last = detection
        self._gap = 0
        self.trail.append(detection.center_px)
        return TrackerResult(detection, self._track_id, False)

    def _miss(self) -> TrackerResult:
        if self._last is None:
            return TrackerResult(None, self._track_id, False)
        self._gap += 1
        if self._gap > self.max_gap_frames:
            self._last = None
            self._gap = 0
            self.trail.clear()
            ended_id = self._track_id
            self._track_id += 1
            return TrackerResult(None, ended_id, True)
        return TrackerResult(None, self._track_id, False)
