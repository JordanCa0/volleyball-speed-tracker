import pytest

from tracker import Detection, Tracker

FPS = 30.0


def det(i, x, y, r=10.0):
    return Detection(i, i / FPS, (x, y), r)


def make_tracker(**kw):
    defaults = dict(max_speed_px_per_s=1000.0, max_gap_frames=5, slack_px=20.0)
    defaults.update(kw)
    return Tracker(**defaults)


def test_smooth_motion_all_accepted():
    tr = make_tracker()
    for i in range(30):
        res = tr.update(det(i, 50 + i * 5, 200))
        assert res.accepted is not None
        assert res.track_id == 0
        assert not res.track_ended


def test_teleport_rejected_as_miss():
    tr = make_tracker()
    tr.update(det(0, 100, 200))
    tr.update(det(1, 105, 200))
    # 800 px in one frame at max 1000 px/s (~33 px/frame + slack): implausible
    res = tr.update(det(2, 905, 200))
    assert res.accepted is None
    # the real ball reappears near where it was: accepted, same track
    res = tr.update(det(3, 115, 200))
    assert res.accepted is not None
    assert res.track_id == 0


def test_gap_tolerated_up_to_limit():
    tr = make_tracker()
    tr.update(det(0, 100, 200))
    for i in range(1, 6):   # 5 misses: track stays alive
        res = tr.update(None)
        assert not res.track_ended
    # detection within plausible range of the coasted gap still joins the track
    res = tr.update(det(6, 140, 200))
    assert res.accepted is not None
    assert res.track_id == 0


def test_track_ends_after_gap_limit():
    tr = make_tracker()
    tr.update(det(0, 100, 200))
    for _ in range(5):
        tr.update(None)
    res = tr.update(None)   # 6th miss: over the limit
    assert res.track_ended
    # next detection starts a fresh track, accepted unconditionally
    res = tr.update(det(10, 900, 50))
    assert res.accepted is not None
    assert res.track_id == 1


def test_trail_follows_track_and_clears_on_end():
    tr = make_tracker()
    for i in range(10):
        tr.update(det(i, 50 + i * 5, 200))
    assert len(tr.trail) == 10
    for _ in range(6):
        tr.update(None)
    assert len(tr.trail) == 0


def test_update_multi_prefers_continuity_over_ranking():
    """A big same-colored distractor must not steal an established track."""
    tr = make_tracker()
    tr.update(det(0, 100, 200))
    tr.update(det(1, 110, 200))
    # candidate list ordered "ball-likeness first": distractor ranks top but
    # sits far away; the real ball continues the track.
    distractor = det(2, 900, 700, r=25.0)
    real = det(2, 120, 200, r=10.0)
    res = tr.update_multi([distractor, real])
    assert res.accepted is not None
    assert res.accepted.center_px == (120, 200)


def test_update_multi_falls_back_to_best_ranked_when_no_track():
    tr = make_tracker()
    best, other = det(0, 400, 300), det(0, 900, 700)
    res = tr.update_multi([best, other])
    assert res.accepted.center_px == (400, 300)


def test_update_multi_no_candidates_is_a_miss():
    tr = make_tracker()
    tr.update(det(0, 100, 200))
    res = tr.update_multi([])
    assert res.accepted is None
