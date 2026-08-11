"""
Speed measurement and hit segmentation (TDD §6.4–6.5).

Two speed outputs with different jobs:
- live readout: finite difference between consecutive detections,
  median-of-3 smoothed (immediate, jittery);
- per-hit peak: quadratic fit to x(t), y(t) over the flight segment,
  differentiated — avoids the max-of-noisy-samples bias.

A "hit" is an uninterrupted ballistic flight segment: opens above a speed
threshold, splits on velocity discontinuities (contacts: hand, block,
floor), closes when the ball slows or the track is lost.
"""
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Hit:
    start_time: float
    end_time: float
    peak_speed_kmh: float
    samples: list[float] = field(default_factory=list)  # instantaneous km/h


class SpeedEngine:
    def __init__(self, meters_per_px: float,
                 open_threshold_kmh: float = 15.0,
                 close_threshold_kmh: float | None = None,
                 split_angle_deg: float = 45.0,
                 split_speed_ratio: float = 2.5,
                 min_turn_speed_kmh: float = 8.0,
                 min_fit_points: int = 5):
        self.mpp = meters_per_px
        self.open_kmh = open_threshold_kmh
        self.close_kmh = close_threshold_kmh if close_threshold_kmh is not None \
            else open_threshold_kmh * 0.8
        self.split_angle_deg = split_angle_deg
        self.split_speed_ratio = split_speed_ratio
        self.min_turn_speed_kmh = min_turn_speed_kmh
        self.min_fit_points = min_fit_points

        self.hits: list[Hit] = []
        self._recent = deque(maxlen=3)          # for the smoothed live readout
        self._last = None                        # (t, x, y) of previous detection
        self._last_v = None                      # (vx, vy) px/s
        self._last_speed_kmh = None
        self._track_id = None
        self._seg: list[tuple[float, float, float]] = []   # (t, x, y), open segment
        self._seg_speeds: list[float] = []

    def update(self, detection, track_id: int, track_ended: bool = False) -> float | None:
        """Feed one tracker result; returns smoothed live speed (km/h) or None."""
        if track_ended or (self._track_id is not None and track_id != self._track_id):
            self._finalize()
            self._last = self._last_v = self._last_speed_kmh = None
        self._track_id = track_id

        if detection is None:
            return None
        t = detection.timestamp
        x, y = detection.center_px

        if self._last is None:
            self._last = (t, x, y)
            return None
        lt, lx, ly = self._last
        dt = t - lt
        if dt <= 0:
            return None

        vx, vy = (x - lx) / dt, (y - ly) / dt
        speed_kmh = math.hypot(vx, vy) * self.mpp * 3.6

        if self._seg and self._is_discontinuity(vx, vy, speed_kmh):
            # Contact (hand/block/floor): close this flight, start the next.
            self._finalize()
        if not self._seg:
            if speed_kmh >= self.open_kmh:
                self._seg = [(lt, lx, ly), (t, x, y)]
                self._seg_speeds = [speed_kmh]
        else:
            self._seg.append((t, x, y))
            self._seg_speeds.append(speed_kmh)
            if speed_kmh < self.close_kmh:
                self._finalize()

        self._last = (t, x, y)
        self._last_v = (vx, vy)
        self._last_speed_kmh = speed_kmh
        self._recent.append(speed_kmh)
        return float(np.median(self._recent))

    def finish(self):
        """Call at end of stream to close any open segment."""
        self._finalize()

    @property
    def last_hit(self) -> Hit | None:
        return self.hits[-1] if self.hits else None

    def _is_discontinuity(self, vx, vy, speed_kmh) -> bool:
        if self._last_v is None or self._last_speed_kmh is None:
            return False
        if speed_kmh > self.split_speed_ratio * self._last_speed_kmh \
                and speed_kmh >= self.open_kmh:
            return True
        if speed_kmh < self.min_turn_speed_kmh \
                or self._last_speed_kmh < self.min_turn_speed_kmh:
            return False   # direction is meaningless at near-zero speed
        lvx, lvy = self._last_v
        dot = vx * lvx + vy * lvy
        norms = math.hypot(vx, vy) * math.hypot(lvx, lvy)
        if norms == 0:
            return False
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / norms))))
        return angle > self.split_angle_deg

    def _finalize(self):
        seg, speeds = self._seg, self._seg_speeds
        self._seg, self._seg_speeds = [], []
        if len(seg) < self.min_fit_points:
            return
        ts = np.array([p[0] for p in seg])
        xs = np.array([p[1] for p in seg])
        ys = np.array([p[2] for p in seg])
        tt = ts - ts[0]   # normalize for fit conditioning
        fx = np.polyfit(tt, xs, 2)
        fy = np.polyfit(tt, ys, 2)
        # Speed of the fitted curve, sampled densely over the segment.
        sample_t = np.linspace(0, tt[-1], 100)
        vx = np.polyval(np.polyder(fx), sample_t)
        vy = np.polyval(np.polyder(fy), sample_t)
        peak = float(np.max(np.hypot(vx, vy)) * self.mpp * 3.6)
        self.hits.append(Hit(start_time=float(ts[0]), end_time=float(ts[-1]),
                             peak_speed_kmh=peak, samples=list(speeds)))
