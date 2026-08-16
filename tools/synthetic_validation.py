"""End-to-end validation against a known trajectory.

The gravity drop test (TDD §8) needs a camera. This is the equivalent we can
run from a desk: composite a *real* volleyball, cut from real footage, onto a
real gym background, moving along a trajectory we choose exactly. Then run the
entire production pipeline over the result and compare.

It exercises the whole chain — detection, radius estimation, depth from
apparent size, the 3D fit, peak extraction — against a number we know to the
millimetre. What it does not reproduce is real motion blur or a real lens, so
it validates the maths and the plumbing, not the optics.

    python tools/synthetic_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import analyse_video
from calibration import CameraIntrinsics

ROOT = Path("/Users/jordan/Projects/volleyball-speed-tracker")
SOURCE = ROOT / "input_videos" / "clip2.mp4"
BALL_FRAME, BALL_XY, BALL_SIDE = 1030, (1350, 668), 26
BACKGROUND_FRAME = 300
OUT = Path("/Users/jordan/.claude/jobs/02e26a43/tmp/synthetic.mp4")

INTRINSICS = CameraIntrinsics(focal_px=1300.0, principal_x=960.0,
                              principal_y=540.0)
FPS = 30.0
G = 9.81


def grab(frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(SOURCE))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame_index}")
    return frame


def ball_sprite() -> tuple[np.ndarray, np.ndarray]:
    """A real ball cut out of real footage, with a circular alpha mask."""
    frame = grab(BALL_FRAME)
    x, y = BALL_XY
    half = BALL_SIDE // 2 + 2
    patch = frame[y - half:y + half, x - half:x + half].copy()
    size = patch.shape[0]
    mask = np.zeros((size, size), np.float32)
    cv2.circle(mask, (size // 2, size // 2), size // 2 - 1, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return patch, mask


def paste(canvas: np.ndarray, patch: np.ndarray, mask: np.ndarray,
          centre: tuple[float, float], diameter_px: float) -> None:
    size = max(6, int(round(diameter_px)))
    p = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)[..., None]
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = x1 + size, y1 + size
    h, w = canvas.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return
    region = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = (region * (1 - m) + p.astype(np.float32) * m
                            ).astype(np.uint8)


def trajectory(n_frames: int) -> list[tuple[float, float, float]]:
    """A serve flying away from the camera, with gravity and some drift."""
    x0, y0, z0 = 0.4, -1.2, 5.0
    vx, vy, vz = 1.5, -1.0, 18.0        # |v| = 18.1 m/s = 65.2 km/h
    points = []
    for i in range(n_frames):
        t = i / FPS
        points.append((x0 + vx * t,
                       y0 + vy * t + 0.5 * G * t * t,
                       z0 + vz * t))
    return points, (vx, vy, vz)


def true_peak_kmh(velocity: tuple[float, float, float], n_frames: int) -> float:
    """Gravity accelerates Y, so the fastest instant is the last one."""
    vx, vy, vz = velocity
    t_end = (n_frames - 1) / FPS
    vy_end = vy + G * t_end
    return float(np.sqrt(vx * vx + vy_end * vy_end + vz * vz) * 3.6)


def main() -> int:
    patch, mask = ball_sprite()
    background = grab(BACKGROUND_FRAME)
    h, w = background.shape[:2]

    n_frames = 30
    points, velocity = trajectory(n_frames)

    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (w, h))
    placed = 0
    for (x, y, z) in points:
        canvas = background.copy()
        u = INTRINSICS.principal_x + x * INTRINSICS.focal_px / z
        v = INTRINSICS.principal_y + y * INTRINSICS.focal_px / z
        diameter = INTRINSICS.focal_px * INTRINSICS.ball_diameter_m / z
        paste(canvas, patch, mask, (u, v), diameter)
        writer.write(canvas)
        placed += 1
    writer.release()

    first, last = points[0], points[-1]
    print(f"synthetic clip: {placed} frames -> {OUT}")
    print(f"  ball travels {first[2]:.1f}m -> {last[2]:.1f}m deep")
    print(f"  apparent diameter "
          f"{INTRINSICS.focal_px * 0.21 / first[2]:.1f}px -> "
          f"{INTRINSICS.focal_px * 0.21 / last[2]:.1f}px")

    expected = true_peak_kmh(velocity, n_frames)
    print(f"  TRUE peak speed: {expected:.1f} km/h")

    print("\nrunning the production pipeline...")
    result = analyse_video(str(OUT), intrinsics=INTRINSICS,
                           ball_model=str(ROOT / "models/volleyball_ball.pt"),
                           confidence=0.25, imgsz=1024)
    print(result.summary())

    if not result.hits:
        print("\nFAIL: no flight segments detected")
        return 1
    best = max(result.hits, key=lambda hit: hit.n_points)
    error = 100 * (best.peak_speed_kmh - expected) / expected
    print(f"\nMEASURED peak: {best.peak_speed_kmh:.1f} km/h "
          f"({best.n_points} points, mean depth {best.mean_depth_m:.1f}m)")
    print(f"TRUE peak:     {expected:.1f} km/h")
    print(f"error:         {error:+.1f}%")
    ok = abs(error) <= 10.0
    print("PASS: within the +-10% target" if ok else "FAIL: outside +-10%")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
