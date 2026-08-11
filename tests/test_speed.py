import math

import numpy as np
import pytest

from speed import SpeedEngine
from tracker import Detection

G = 9.81


def feed(engine, positions_m, fps, mpp, track_id=0):
    """positions_m: list of (x, y) in meters; converted to px via mpp."""
    for i, (x, y) in enumerate(positions_m):
        d = Detection(i, i / fps, (x / mpp, y / mpp), 10.0)
        engine.update(d, track_id)
    engine.finish()


def test_free_fall_peak_matches_gravity():
    """Gravity is the ground truth: v = g*t, exactly (TDD §8 drop test)."""
    fps, mpp, duration = 60.0, 0.01, 1.2
    rng = np.random.default_rng(42)
    positions = []
    for i in range(int(duration * fps) + 1):
        t = i / fps
        noise = rng.normal(0, 0.5) * mpp   # ±0.5 px detection jitter
        positions.append((2.0, 0.5 * G * t * t + noise))
    engine = SpeedEngine(mpp)
    feed(engine, positions, fps, mpp)

    assert len(engine.hits) == 1
    true_final_kmh = G * duration * 3.6          # 42.4 km/h
    assert engine.hits[0].peak_speed_kmh == pytest.approx(true_final_kmh, rel=0.05)


def test_constant_velocity_peak():
    fps, mpp, speed_ms = 30.0, 0.02, 10.0        # 36 km/h
    positions = [(speed_ms * i / fps, 1.0) for i in range(40)]
    engine = SpeedEngine(mpp)
    feed(engine, positions, fps, mpp)

    assert len(engine.hits) == 1
    assert engine.hits[0].peak_speed_kmh == pytest.approx(36.0, rel=0.02)


def test_slow_motion_produces_no_hits():
    fps, mpp = 30.0, 0.02
    positions = [(0.5 * i / fps, 1.0) for i in range(60)]   # 0.5 m/s = 1.8 km/h
    engine = SpeedEngine(mpp)
    feed(engine, positions, fps, mpp)
    assert engine.hits == []


def test_bounce_splits_into_two_hits():
    """A sharp direction change (contact) must split the segment."""
    fps, mpp, speed_ms = 60.0, 0.01, 8.0
    positions = []
    n = 30
    for i in range(n):                            # traveling down-right
        t = i / fps
        positions.append((speed_ms * t * 0.707, speed_ms * t * 0.707))
    apex_x, apex_y = positions[-1]
    for i in range(1, n):                         # deflected: up-right
        t = i / fps
        positions.append((apex_x + speed_ms * t * 0.707, apex_y - speed_ms * t * 0.707))
    engine = SpeedEngine(mpp)
    feed(engine, positions, fps, mpp)

    assert len(engine.hits) == 2
    for hit in engine.hits:
        assert hit.peak_speed_kmh == pytest.approx(8.0 * 3.6, rel=0.05)


def test_track_change_finalizes_open_segment():
    fps, mpp, speed_ms = 30.0, 0.02, 10.0
    engine = SpeedEngine(mpp)
    for i in range(20):
        d = Detection(i, i / fps, (speed_ms * i / fps / mpp, 100.0), 10.0)
        engine.update(d, track_id=0)
    # ball lost; a new track starts elsewhere
    engine.update(Detection(40, 40 / fps, (5000.0, 5000.0), 10.0),
                  track_id=1, track_ended=False)
    assert len(engine.hits) == 1
    assert engine.hits[0].peak_speed_kmh == pytest.approx(36.0, rel=0.05)


def test_noisy_peak_not_inflated():
    """Trajectory fit must not report true speed + luckiest noise spike."""
    fps, mpp, speed_ms = 30.0, 0.02, 15.0        # 54 km/h
    rng = np.random.default_rng(7)
    positions = [(speed_ms * i / fps + rng.normal(0, 1.5) * mpp, 1.0)
                 for i in range(45)]
    engine = SpeedEngine(mpp)
    feed(engine, positions, fps, mpp)

    assert len(engine.hits) >= 1
    hit = max(engine.hits, key=lambda h: len(h.samples))
    # the fitted peak must sit far closer to truth than the raw
    # finite-difference max, which noise inflates
    raw_error = max(hit.samples) - 54.0
    fit_error = abs(hit.peak_speed_kmh - 54.0)
    assert raw_error > 3 * fit_error
    assert hit.peak_speed_kmh == pytest.approx(54.0, rel=0.07)
