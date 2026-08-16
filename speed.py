"""Speed measurement and hit segmentation (TDD §6.4–6.5).

The camera films from behind the server, so the ball's motion is mostly along
the optical axis. Pixel displacement barely registers there, which is why
everything here works in **metres**: each observation is lifted to a 3D point
using the ball's apparent size as a depth cue (see calibration.py), and speed
is the magnitude of the derivative of a fitted 3D trajectory.

Two speed outputs with different jobs:
- live readout: finite difference between consecutive samples, median-of-3
  smoothed (immediate, jittery);
- per-hit peak: quadratic fit to X(t), Y(t), Z(t) over the flight segment,
  differentiated — avoids the max-of-noisy-samples bias.

A "hit" is an uninterrupted ballistic flight segment: opens above a speed
threshold, splits on velocity discontinuities (contacts: hand, block, floor),
closes when the ball slows, leaves frame, or the track is lost.

Depth is the noisy axis. A 3% error in apparent radius is a 3% error in depth,
which at 10 m is 30 cm — comparable to how far the ball travels between frames
at 30 fps. Two things keep that from wrecking the result:

- radius is median-smoothed before it becomes depth, killing single-frame
  spikes without lagging a real trend;
- segmentation decisions (open, close, split) use **image-plane** motion,
  which is precise, while only the reported speed uses depth. A noisy Z must
  not be allowed to fabricate a "contact".
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from calibration import CameraIntrinsics


@dataclass
class BallSample:
    """One observation of the ball. Duck-typed by the tracker output."""

    frame_idx: int
    timestamp: float
    center_px: tuple[float, float]
    radius_px: float
    confidence: float = 1.0


@dataclass
class Hit:
    start_time: float
    end_time: float
    peak_speed_kmh: float
    samples: list[float] = field(default_factory=list)   # instantaneous km/h
    n_points: int = 0
    mean_depth_m: float = 0.0
    truncated: bool = False      # ball left frame / track lost mid-flight

    @property
    def reliable(self) -> bool:
        """Enough of an arc to trust the fit.

        A quadratic through four points will fit anything; the peak it reports
        is then a property of the noise, not the ball.
        """
        return self.n_points >= 8 and not self.truncated


class SpeedEngine:
    def __init__(self, intrinsics: CameraIntrinsics,
                 open_threshold_kmh: float = 15.0,
                 close_threshold_kmh: float | None = None,
                 split_angle_deg: float = 45.0,
                 split_speed_ratio: float = 2.5,
                 min_turn_speed_kmh: float = 8.0,
                 min_fit_points: int = 5,
                 max_speed_kmh: float = 160.0,
                 radius_smoothing: int = 3,
                 direction_baseline: int = 3,
                 max_gap_frames: int = 5):
        self.intrinsics = intrinsics
        self.open_kmh = open_threshold_kmh
        self.close_kmh = (close_threshold_kmh if close_threshold_kmh is not None
                          else open_threshold_kmh * 0.8)
        self.split_angle_deg = split_angle_deg
        self.split_speed_ratio = split_speed_ratio
        self.min_turn_speed_kmh = min_turn_speed_kmh
        self.min_fit_points = min_fit_points
        self.max_speed_kmh = max_speed_kmh

        self.max_gap_frames = max_gap_frames
        self.hits: list[Hit] = []
        self._radii: deque[float] = deque(maxlen=max(1, radius_smoothing))
        # Direction is judged over a multi-frame baseline, not frame to frame.
        # A ±2px box jitter swings the single-frame direction wildly when the
        # ball only moves a few pixels per frame, and every one of those swings
        # looked like a contact — on synthetic footage with a known-continuous
        # arc it fragmented one flight into three.
        self.direction_baseline = max(1, direction_baseline)
        self._px_trail: deque[tuple[float, float, float]] = deque(
            maxlen=2 * self.direction_baseline + 1)
        self._misses = 0
        self._recent = deque(maxlen=3)       # smoothed live readout
        self._last = None                    # (t, X, Y, Z)
        self._last_px = None                 # (u, v) for image-plane decisions
        self._last_px_speed = None           # |image-plane velocity|, px/s
        self._last_speed_kmh = None
        self._track_id = None
        self._seg: list[tuple[float, float, float, float]] = []   # (t,X,Y,Z)
        self._seg_speeds: list[float] = []
        self._pending_truncated = False

    # ---------------------------------------------------------------- public

    def update(self, sample: BallSample | None, track_id: int = 0,
               track_ended: bool = False,
               left_frame: bool = False) -> float | None:
        """Feed one tracker result; returns smoothed live speed (km/h) or None."""
        if track_ended or left_frame or \
                (self._track_id is not None and track_id != self._track_id):
            self._pending_truncated = bool(self._seg) and (track_ended or left_frame)
            self._finalize()
            self._reset_motion()
        self._track_id = track_id

        if sample is None or sample.radius_px <= 0:
            # A gap must end the flight rather than be bridged. Interpolating
            # across unobserved frames would invent a straight line through
            # whatever actually happened and feed it to the fit.
            self._misses += 1
            if self._misses >= self.max_gap_frames and self._seg:
                self._pending_truncated = True
                self._finalize()
                self._reset_motion()
            return None

        self._misses = 0
        self._radii.append(sample.radius_px)
        radius = float(np.median(self._radii))
        t = sample.timestamp
        x, y, z = self.intrinsics.to_world(sample.center_px, radius)
        u, v = sample.center_px

        if self._last is None:
            self._last, self._last_px = (t, x, y, z), (u, v)
            return None

        lt, lx, ly, lz = self._last
        dt = t - lt
        if dt <= 0:
            return None

        vx, vy, vz = (x - lx) / dt, (y - ly) / dt, (z - lz) / dt
        speed_kmh = math.sqrt(vx * vx + vy * vy + vz * vz) * 3.6

        # Image-plane motion drives the segmentation decisions (§ module doc).
        self._px_trail.append((t, u, v))
        lu, lv = self._last_px
        vu, vv = (u - lu) / dt, (v - lv) / dt

        if speed_kmh > self.max_speed_kmh:
            # Physically impossible: a bad radius or a false positive. Drop the
            # sample rather than let it open or split a segment.
            self._last, self._last_px = (t, x, y, z), (u, v)
            return None

        # Depth noise must not reach the segmentation decisions. The
        # instantaneous 3D speed inherits every radius wobble — a 1px error on
        # an 8px ball swings the depth by 12% — and on synthetic footage with a
        # known-continuous arc that was enough to trip the "speed jumped, so
        # something hit it" rule twice, fragmenting one flight into three.
        # Open/close therefore run on a median-smoothed speed, and the jump
        # test runs on image-plane motion, which carries no depth at all.
        self._recent.append(speed_kmh)
        smoothed_kmh = float(np.median(self._recent))
        px_speed = math.hypot(vu, vv)

        if self._seg and self._is_discontinuity(px_speed, smoothed_kmh):
            self._finalize()

        if not self._seg:
            if smoothed_kmh >= self.open_kmh:
                self._seg = [(lt, lx, ly, lz), (t, x, y, z)]
                self._seg_speeds = [speed_kmh]
        else:
            self._seg.append((t, x, y, z))
            self._seg_speeds.append(speed_kmh)
            if smoothed_kmh < self.close_kmh:
                self._finalize()

        self._last, self._last_px = (t, x, y, z), (u, v)
        self._last_px_speed = px_speed
        self._last_speed_kmh = smoothed_kmh
        return smoothed_kmh

    def finish(self) -> None:
        """Close any open segment at end of stream."""
        self._finalize()

    @property
    def last_hit(self) -> Hit | None:
        return self.hits[-1] if self.hits else None

    # --------------------------------------------------------------- private

    def _reset_motion(self) -> None:
        self._last = self._last_px = self._last_px_speed = None
        self._last_speed_kmh = None
        self._radii.clear()
        self._px_trail.clear()
        self._misses = 0

    def _direction_change_deg(self) -> float | None:
        """Angle between two *non-overlapping* windows of image motion.

        Overlapping windows were tried and do not work: sharing frames blunts a
        real contact into a gradual turn, and a genuine bounce stopped
        registering at all. Non-overlapping windows keep the contact sharp
        while still averaging jitter away inside each one. The cost is that a
        contact is recognised `direction_baseline` frames late.
        """
        n = self.direction_baseline
        if len(self._px_trail) < 2 * n + 1:
            return None
        _, u0, v0 = self._px_trail[0]
        _, u1, v1 = self._px_trail[n]
        _, u2, v2 = self._px_trail[2 * n]
        ax, ay = u1 - u0, v1 - v0
        bx, by = u2 - u1, v2 - v1
        # Below a few pixels of travel the direction is noise, not motion.
        if math.hypot(ax, ay) < 3.0 or math.hypot(bx, by) < 3.0:
            return None
        cos = ((ax * bx + ay * by)
               / (math.hypot(ax, ay) * math.hypot(bx, by)))
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    def _is_discontinuity(self, px_speed: float, smoothed_kmh: float) -> bool:
        if self._last_px_speed is None or self._last_speed_kmh is None:
            return False
        # A contact shows up as a jump in how fast the ball crosses the image.
        if px_speed > self.split_speed_ratio * max(self._last_px_speed, 1e-6) \
                and smoothed_kmh >= self.open_kmh:
            return True
        if smoothed_kmh < self.min_turn_speed_kmh:
            return False   # direction is meaningless at near-zero speed
        angle = self._direction_change_deg()
        return angle is not None and angle > self.split_angle_deg

    def _finalize(self) -> None:
        seg, speeds = self._seg, self._seg_speeds
        truncated, self._pending_truncated = self._pending_truncated, False
        self._seg, self._seg_speeds = [], []
        if len(seg) < self.min_fit_points:
            return

        ts = np.array([p[0] for p in seg])
        tt = ts - ts[0]                       # normalise for fit conditioning
        coords = [np.array([p[i] for p in seg]) for i in (1, 2, 3)]

        # Quadratic per axis: in-flight paths are near-ballistic, and drag over
        # a single arc is gentle enough that the curvature term absorbs it.
        derivatives = []
        sample_t = np.linspace(0, tt[-1], 100)
        for values in coords:
            fit = np.polyfit(tt, values, 2)
            derivatives.append(np.polyval(np.polyder(fit), sample_t))

        speed_ms = np.sqrt(sum(d * d for d in derivatives))
        peak = float(np.max(speed_ms) * 3.6)
        if peak > self.max_speed_kmh:
            peak = float(np.median(speed_ms) * 3.6)

        self.hits.append(Hit(
            start_time=float(ts[0]),
            end_time=float(ts[-1]),
            peak_speed_kmh=peak,
            samples=list(speeds),
            n_points=len(seg),
            mean_depth_m=float(np.mean(coords[2])),
            truncated=truncated,
        ))
