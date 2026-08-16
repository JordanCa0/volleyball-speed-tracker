"""Speed engine tests, in 3D.

Everything is specified in metres and projected into the image the way a real
camera would, so the tests exercise the depth-from-size path rather than
assuming it.
"""
import numpy as np
import pytest

from calibration import CameraIntrinsics
from speed import BallSample, SpeedEngine

G = 9.81
INTR = CameraIntrinsics(focal_px=1300.0, principal_x=960.0, principal_y=540.0)


def project(point_m, intrinsics=INTR, radius_noise=0.0, rng=None):
    """(X, Y, Z) in metres -> the (center_px, radius_px) a camera would see."""
    x, y, z = point_m
    u = intrinsics.principal_x + x * intrinsics.focal_px / z
    v = intrinsics.principal_y + y * intrinsics.focal_px / z
    r = intrinsics.focal_px * intrinsics.ball_diameter_m / (2.0 * z)
    if radius_noise and rng is not None:
        r += rng.normal(0, radius_noise)
    return (u, v), r


def feed(engine, points_m, fps, radius_noise=0.0, seed=0, track_id=0):
    rng = np.random.default_rng(seed)
    for i, point in enumerate(points_m):
        center, radius = project(point, engine.intrinsics, radius_noise, rng)
        engine.update(BallSample(i, i / fps, center, radius), track_id)
    engine.finish()


def test_depth_axis_motion_is_measured():
    """The whole point: a ball flying away from the camera.

    Pixel displacement here is nearly zero — the speed lives entirely in the
    changing apparent size. The old scalar-scale engine measured 0 for this.
    """
    fps, speed_ms = 30.0, 10.0            # 36 km/h straight down the barrel
    points = [(0.0, 0.0, 6.0 + speed_ms * i / fps) for i in range(30)]
    engine = SpeedEngine(INTR)
    feed(engine, points, fps)

    assert len(engine.hits) == 1
    assert engine.hits[0].peak_speed_kmh == pytest.approx(36.0, rel=0.05)


def test_free_fall_peak_matches_gravity():
    """Gravity is the ground truth: v = g*t, exactly (TDD §8 drop test)."""
    fps, duration, depth = 60.0, 1.2, 10.0
    points = [(2.0, 0.5 * G * (i / fps) ** 2, depth)
              for i in range(int(duration * fps) + 1)]
    engine = SpeedEngine(INTR)
    feed(engine, points, fps)

    assert len(engine.hits) == 1
    assert engine.hits[0].peak_speed_kmh == pytest.approx(G * duration * 3.6, rel=0.05)


def test_diagonal_motion_combines_all_three_axes():
    fps = 30.0
    vx, vy, vz = 6.0, 2.0, 8.0            # |v| = 10.2 m/s = 36.8 km/h
    points = [(vx * i / fps, vy * i / fps, 7.0 + vz * i / fps) for i in range(30)]
    engine = SpeedEngine(INTR)
    feed(engine, points, fps)

    expected = np.hypot(np.hypot(vx, vy), vz) * 3.6
    assert len(engine.hits) == 1
    assert engine.hits[0].peak_speed_kmh == pytest.approx(expected, rel=0.05)


def test_slow_motion_produces_no_hits():
    fps = 30.0
    points = [(0.5 * i / fps, 1.0, 8.0) for i in range(60)]   # 1.8 km/h
    engine = SpeedEngine(INTR)
    feed(engine, points, fps)
    assert engine.hits == []


def test_bounce_splits_into_two_hits():
    """A sharp direction change (contact) must split the segment."""
    fps, speed = 60.0, 8.0
    points, n, depth = [], 30, 9.0
    for i in range(n):
        t = i / fps
        points.append((speed * t * 0.707, speed * t * 0.707, depth))
    ax, ay, _ = points[-1]
    for i in range(1, n):
        t = i / fps
        points.append((ax + speed * t * 0.707, ay - speed * t * 0.707, depth))

    engine = SpeedEngine(INTR)
    feed(engine, points, fps)

    assert len(engine.hits) == 2
    for hit in engine.hits:
        assert hit.peak_speed_kmh == pytest.approx(speed * 3.6, rel=0.06)


def test_leaving_frame_closes_and_flags_the_segment():
    fps, speed_ms = 30.0, 12.0
    engine = SpeedEngine(INTR)
    for i in range(15):
        center, radius = project((speed_ms * i / fps, 0.0, 8.0))
        engine.update(BallSample(i, i / fps, center, radius), left_frame=(i == 14))
    engine.finish()

    assert len(engine.hits) == 1
    assert engine.hits[0].truncated
    assert not engine.hits[0].reliable, "a truncated flight must not read as reliable"


def test_short_segment_is_not_reliable():
    fps, speed_ms = 30.0, 12.0
    points = [(speed_ms * i / fps, 0.0, 8.0) for i in range(6)]
    engine = SpeedEngine(INTR)
    feed(engine, points, fps)
    assert engine.hits and not engine.hits[0].reliable


def test_radius_noise_does_not_inflate_the_peak():
    """3% radius noise is realistic; it must not become a fake fast serve."""
    fps, speed_ms, depth = 30.0, 20.0, 9.0
    true_radius = INTR.focal_px * INTR.ball_diameter_m / (2 * depth)
    points = [(speed_ms * i / fps, 0.0, depth) for i in range(30)]
    engine = SpeedEngine(INTR)
    feed(engine, points, fps, radius_noise=0.032 * true_radius, seed=11)

    assert engine.hits
    hit = max(engine.hits, key=lambda h: h.n_points)
    assert hit.peak_speed_kmh == pytest.approx(speed_ms * 3.6, rel=0.15)


def test_impossible_speed_is_rejected():
    """A blown radius estimate must not produce a 900 km/h serve."""
    fps = 30.0
    engine = SpeedEngine(INTR, max_speed_kmh=160.0)
    for i in range(10):
        center, radius = project((0.0, 0.0, 8.0))
        engine.update(BallSample(i, i / fps, center, radius))
    # One frame where the detector reports a wildly wrong (tiny) ball.
    engine.update(BallSample(10, 10 / fps, (960.0, 540.0), 1.0))
    engine.finish()

    for hit in engine.hits:
        assert hit.peak_speed_kmh <= 160.0


def test_track_change_finalizes_open_segment():
    fps, speed_ms = 30.0, 10.0
    engine = SpeedEngine(INTR)
    for i in range(20):
        center, radius = project((speed_ms * i / fps, 0.0, 8.0))
        engine.update(BallSample(i, i / fps, center, radius), track_id=0)
    center, radius = project((0.0, 0.0, 20.0))
    engine.update(BallSample(40, 40 / fps, center, radius), track_id=1)

    assert len(engine.hits) == 1
    assert engine.hits[0].peak_speed_kmh == pytest.approx(36.0, rel=0.06)
