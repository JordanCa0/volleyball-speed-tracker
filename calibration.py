"""Camera intrinsics and depth from the ball's known size (TDD §6.3).

The camera films from behind the server, so the ball flies away from the lens
and its motion is mostly along the optical axis. Pixel displacement barely
registers there — but apparent *size* does, and a volleyball is a known
21 cm. That makes the ball its own ruler:

    Z = f_px * ball_diameter_m / diameter_px

which is a real distance in metres, per frame, with no court markings and no
assumption about which lane the ball flies down.

Calibrating `f_px` needs one reference: the ball at a measured distance.
Because depth is `f*D/w`, any constant bias in how we measure `w` cancels
exactly, *provided calibration and measurement use the same estimator*. They
do — both go through `RadiusEstimator` in tracking/detectors.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

BALL_DIAMETER_M = 0.21


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics, in pixels, plus the ball size they were fit with."""

    focal_px: float
    principal_x: float
    principal_y: float
    ball_diameter_m: float = BALL_DIAMETER_M

    def __post_init__(self) -> None:
        if self.focal_px <= 0:
            raise ValueError(f"focal_px must be positive, got {self.focal_px}")
        if self.ball_diameter_m <= 0:
            raise ValueError("ball_diameter_m must be positive")

    def depth_m(self, radius_px: float) -> float:
        """Distance to a ball whose apparent radius is `radius_px`."""
        if radius_px <= 0:
            raise ValueError(f"radius_px must be positive, got {radius_px}")
        return self.focal_px * self.ball_diameter_m / (2.0 * radius_px)

    def to_world(self, center_px: tuple[float, float],
                 radius_px: float) -> tuple[float, float, float]:
        """Image observation -> (X, Y, Z) metres in the camera frame.

        X is right, Y is down, Z is into the scene. The origin is the camera,
        so these are directly differentiable into a velocity.
        """
        z = self.depth_m(radius_px)
        u, v = center_px
        x = (u - self.principal_x) * z / self.focal_px
        y = (v - self.principal_y) * z / self.focal_px
        return x, y, z

    def as_dict(self) -> dict:
        return {"focal_px": self.focal_px,
                "principal_x": self.principal_x,
                "principal_y": self.principal_y,
                "ball_diameter_m": self.ball_diameter_m}


def focal_length_from_reference(radius_px: float, distance_m: float,
                                frame_wh: tuple[int, int],
                                ball_diameter_m: float = BALL_DIAMETER_M
                                ) -> CameraIntrinsics:
    """Fit `f_px` from the ball held at a measured distance.

    Inverts Z = f*D/(2r). The principal point is assumed to be the frame
    centre, which is accurate enough for consumer cameras — an offset of a few
    pixels moves the lateral estimate by millimetres at these distances.
    """
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    width, height = frame_wh
    focal_px = 2.0 * radius_px * distance_m / ball_diameter_m
    return CameraIntrinsics(focal_px=focal_px,
                            principal_x=width / 2.0,
                            principal_y=height / 2.0,
                            ball_diameter_m=ball_diameter_m)


def focal_length_from_fov(frame_wh: tuple[int, int],
                          horizontal_fov_deg: float,
                          ball_diameter_m: float = BALL_DIAMETER_M
                          ) -> CameraIntrinsics:
    """Estimate intrinsics from a quoted field of view.

    A fallback for when no reference clip exists. Phone main cameras sit
    around 70° horizontal. Less trustworthy than a reference measurement —
    quoted FOVs are marketing figures and often include crop factors — so the
    setup flow should prefer `focal_length_from_reference`.
    """
    if not 10.0 < horizontal_fov_deg < 170.0:
        raise ValueError("horizontal_fov_deg outside a plausible range")
    width, height = frame_wh
    focal_px = (width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)
    return CameraIntrinsics(focal_px=focal_px,
                            principal_x=width / 2.0,
                            principal_y=height / 2.0,
                            ball_diameter_m=ball_diameter_m)


def save_to_config(intrinsics: CameraIntrinsics, config: dict) -> dict:
    config["intrinsics"] = intrinsics.as_dict()
    return config


def load_from_config(config: dict) -> CameraIntrinsics | None:
    data = config.get("intrinsics")
    if not data:
        return None
    return CameraIntrinsics(
        focal_px=data["focal_px"],
        principal_x=data["principal_x"],
        principal_y=data["principal_y"],
        ball_diameter_m=data.get("ball_diameter_m", BALL_DIAMETER_M),
    )
