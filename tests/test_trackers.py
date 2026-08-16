"""Ball selection tests.

These encode the failure that made club-gym footage unusable: a static false
positive (a court line, a sponsor banner) outranking the real ball.
"""
import numpy as np
import pytest
import supervision as sv

from tracking.trackers import BallTracker


def detections(*items):
    """Build detections from (x, y, confidence) or (x, y, confidence, radius)."""
    if not items:
        return sv.Detections.empty()
    xyxy, confidence = [], []
    for item in items:
        x, y, conf = item[:3]
        r = item[3] if len(item) > 3 else 10.0
        xyxy.append([x - r, y - r, x + r, y + r])
        confidence.append(conf)
    return sv.Detections(xyxy=np.array(xyxy, dtype=float),
                         confidence=np.array(confidence, dtype=float))


def centre_of(result):
    return result.get_anchors_coordinates(sv.Position.CENTER)[0]


def test_empty_detections_pass_through():
    tracker = BallTracker()
    assert len(tracker.update(detections())) == 0


def test_single_candidate_is_returned():
    tracker = BallTracker()
    result = tracker.update(detections((100, 100, 0.4)))
    assert len(result) == 1
    assert centre_of(result) == pytest.approx([100, 100])


def test_static_false_positive_is_rejected_in_favour_of_the_ball():
    """The bug this strategy exists to fix.

    A banner sits at (500, 200) every frame. The ball flies across at lower
    confidence. The banner must not win once it has proved itself stationary.
    """
    tracker = BallTracker(static_min_hits=6, static_window=25)
    for i in range(10):
        tracker.update(detections((500, 200, 0.9), (100 + i * 30, 400, 0.5)))

    result = tracker.update(detections((500, 200, 0.9), (400, 400, 0.5)))
    assert len(result) == 1
    assert centre_of(result) == pytest.approx([400, 400]), \
        "picked the stationary banner over the moving ball"


def test_receding_ball_is_not_mistaken_for_furniture():
    """Filming from behind the server, the ball shrinks instead of moving.

    Position-only staticness threw away 86% of real detections on end-on
    footage, because a ball flying away holds nearly the same image position
    for its whole flight. Shrinking is what distinguishes it from a banner.
    """
    tracker = BallTracker(static_min_hits=6, static_window=25)
    radius = 22.0
    kept = 0
    for _ in range(20):
        radius *= 0.93                      # receding, ~7% smaller each frame
        result = tracker.update(detections((900, 400, 0.8, radius)))
        kept += len(result)
    assert kept >= 18, f"suppression ate the receding ball ({kept}/20 kept)"


def test_fixed_size_at_fixed_position_is_still_suppressed():
    """The size test must not defeat the original purpose."""
    tracker = BallTracker(static_min_hits=6)
    for _ in range(10):
        tracker.update(detections((500, 200, 0.9, 14.0)))
    assert len(tracker.update(detections((500, 200, 0.9, 14.0)))) == 0


def test_everything_static_reports_nothing():
    """Better to report no ball than to report a scoreboard graphic."""
    tracker = BallTracker(static_min_hits=6)
    for _ in range(10):
        tracker.update(detections((500, 200, 0.9), (700, 300, 0.8)))
    assert len(tracker.update(detections((500, 200, 0.9), (700, 300, 0.8)))) == 0


def test_moving_ball_survives_a_long_history():
    tracker = BallTracker(static_min_hits=6)
    for i in range(30):
        result = tracker.update(detections((50 + i * 25, 300, 0.7)))
        assert len(result) == 1


def test_confidence_breaks_ties_between_moving_candidates():
    tracker = BallTracker()
    result = tracker.update(detections((100, 100, 0.3), (600, 600, 0.85)))
    assert centre_of(result) == pytest.approx([600, 600])


def test_centroid_strategy_still_available():
    tracker = BallTracker(strategy="centroid")
    tracker.update(detections((100, 100, 0.5)))
    result = tracker.update(detections((105, 105, 0.2), (900, 900, 0.9)))
    # The centroid strategy ignores confidence entirely and takes the nearest.
    assert centre_of(result) == pytest.approx([105, 105])


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        BallTracker(strategy="magic")


def test_reset_clears_static_history():
    tracker = BallTracker(static_min_hits=6)
    for _ in range(10):
        tracker.update(detections((500, 200, 0.9)))
    assert len(tracker.update(detections((500, 200, 0.9)))) == 0
    tracker.reset()
    assert len(tracker.update(detections((500, 200, 0.9)))) == 1
