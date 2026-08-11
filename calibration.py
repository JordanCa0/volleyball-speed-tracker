"""
Pixel -> real-world scale calibration (TDD §6.3).

Preferred: two clicked points a known real distance apart (court lines are
standardized: attack line to center line = 3.00 m). Fallback: the ball's
known diameter measured while slow and sharp.
"""
import math
from dataclasses import dataclass

import cv2


@dataclass
class ScaleCalibration:
    method: str            # "two_point" | "ball_diameter"
    meters_per_px: float


def two_point(p1, p2, distance_m: float) -> ScaleCalibration:
    dist_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if dist_px == 0:
        raise ValueError("calibration points are identical")
    return ScaleCalibration("two_point", distance_m / dist_px)


def from_ball_radius(radius_px: float, ball_diameter_m: float = 0.21) -> ScaleCalibration:
    return ScaleCalibration("ball_diameter", ball_diameter_m / (2 * radius_px))


def save_to_config(cal: ScaleCalibration, config: dict) -> dict:
    config["scale"] = {"method": cal.method, "meters_per_px": cal.meters_per_px}
    return config


def load_from_config(config: dict) -> ScaleCalibration | None:
    s = config.get("scale")
    return ScaleCalibration(s["method"], s["meters_per_px"]) if s else None


def interactive_two_point(frame, distance_m: float) -> ScaleCalibration:
    """Show the frame; user clicks two points a known distance apart."""
    window = f"click 2 points {distance_m} m apart (r=redo, ENTER=confirm)"
    points = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = frame.copy()
        for p in points:
            cv2.drawMarker(canvas, p, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        if len(points) == 2:
            cv2.line(canvas, points[0], points[1], (0, 255, 255), 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            points.clear()
        if key in (13, 10) and len(points) == 2:   # ENTER
            break
        if key == ord("q"):
            raise SystemExit("calibration cancelled")
    cv2.destroyWindow(window)
    return two_point(points[0], points[1], distance_m)
