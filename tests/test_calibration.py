"""Intrinsics and depth-from-size tests."""
import pytest

from calibration import (
    BALL_DIAMETER_M,
    CameraIntrinsics,
    focal_length_from_fov,
    focal_length_from_reference,
    load_from_config,
    save_to_config,
)


def test_depth_inverts_the_projection():
    intr = CameraIntrinsics(1300.0, 960.0, 540.0)
    for distance in (2.0, 6.0, 12.0, 20.0):
        radius = intr.focal_px * BALL_DIAMETER_M / (2 * distance)
        assert intr.depth_m(radius) == pytest.approx(distance)


def test_focal_from_reference_round_trips():
    """Calibrate at 5 m, then a ball at 5 m must read back as 5 m."""
    intr = focal_length_from_reference(radius_px=27.3, distance_m=5.0,
                                       frame_wh=(1920, 1080))
    assert intr.depth_m(27.3) == pytest.approx(5.0)
    assert intr.principal_x == 960.0 and intr.principal_y == 540.0


def test_a_constant_bias_in_radius_cancels():
    """The reason we can use the bounding box despite it not being the ball.

    If the estimator reads every radius 40% large, calibrating with the same
    estimator absorbs it exactly — depth comes out right anyway.
    """
    bias = 1.4
    true_distance, true_radius = 8.0, 17.06
    intr = focal_length_from_reference(radius_px=true_radius * bias,
                                       distance_m=true_distance,
                                       frame_wh=(1920, 1080))
    measured_radius = (intr.focal_px * BALL_DIAMETER_M / (2 * 12.0)) * 1.0
    # A ball actually at 12 m, measured with the same biased estimator:
    biased_reading = (intr.focal_px * BALL_DIAMETER_M / (2 * 12.0))
    assert intr.depth_m(biased_reading) == pytest.approx(12.0)
    assert measured_radius == pytest.approx(biased_reading)


def test_to_world_places_the_principal_point_on_the_axis():
    intr = CameraIntrinsics(1300.0, 960.0, 540.0)
    radius = intr.focal_px * BALL_DIAMETER_M / (2 * 10.0)
    x, y, z = intr.to_world((960.0, 540.0), radius)
    assert (x, y) == pytest.approx((0.0, 0.0))
    assert z == pytest.approx(10.0)


def test_to_world_lateral_offset_scales_with_depth():
    """The same pixel offset means more metres when the ball is further away."""
    intr = CameraIntrinsics(1300.0, 960.0, 540.0)
    near_r = intr.focal_px * BALL_DIAMETER_M / (2 * 5.0)
    far_r = intr.focal_px * BALL_DIAMETER_M / (2 * 15.0)
    near_x, _, _ = intr.to_world((1160.0, 540.0), near_r)
    far_x, _, _ = intr.to_world((1160.0, 540.0), far_r)
    assert far_x == pytest.approx(3 * near_x)


def test_fov_fallback_is_plausible():
    intr = focal_length_from_fov((1920, 1080), 70.0)
    assert 1200 < intr.focal_px < 1500


def test_config_round_trip():
    intr = CameraIntrinsics(1234.5, 960.0, 540.0)
    config = save_to_config(intr, {})
    assert load_from_config(config) == intr


def test_missing_config_returns_none():
    assert load_from_config({}) is None


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        CameraIntrinsics(bad, 960.0, 540.0)
    with pytest.raises(ValueError):
        focal_length_from_reference(radius_px=bad, distance_m=5.0,
                                    frame_wh=(1920, 1080))


def test_zero_radius_is_rejected_rather_than_infinite():
    intr = CameraIntrinsics(1300.0, 960.0, 540.0)
    with pytest.raises(ValueError):
        intr.depth_m(0.0)
