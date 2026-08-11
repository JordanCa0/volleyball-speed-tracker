"""
Color-based ball detection (TDD §6.1).

The ball's color is sampled from a user-drawn box (see detect.py, 'b' key)
and per-frame detection finds the largest plausibly-circular blob in that
color range, optionally restricted to moving pixels.
"""
import cv2
import numpy as np


def sample_ball_color(frame, roi):
    """Median HSV of the central half of the selected box -> detection bounds."""
    x, y, w, h = roi
    patch = frame[y + h // 4: y + h - h // 4 or y + h,
                  x + w // 4: x + w - w // 4 or x + w]
    if patch.size == 0:
        patch = frame[y:y + h, x:x + w]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    med = np.median(hsv.reshape(-1, 3), axis=0)
    lower = [int(max(med[0] - 12, 0)), int(max(med[1] - 70, 30)), int(max(med[2] - 70, 30))]
    upper = [int(min(med[0] + 12, 179)), 255, 255]
    return lower, upper


MIN_CIRCULARITY = 0.35


def color_mask(frame, lower, upper, motion_mask=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    if motion_mask is not None:
        # Require ball pixels to also be moving: rejects static same-color
        # objects (broadcast GUI overlays, floors, walls).
        mask = cv2.bitwise_and(mask, motion_mask)
    mask = cv2.erode(mask, None, iterations=1)
    return cv2.dilate(mask, None, iterations=2)


def find_candidates(frame, lower, upper, min_r, max_r, motion_mask=None,
                    min_circularity=MIN_CIRCULARITY, expected_r=None):
    """All plausible ball blobs this frame, best-looking first.

    Returning every candidate (rather than just the largest) lets the
    tracker pick by continuity — essential when the background contains
    large same-colored regions (gym walls, sponsor boards) that would
    otherwise always win on size.
    """
    mask = color_mask(frame, lower, upper, motion_mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        (x, y), r = cv2.minEnclosingCircle(c)
        if not min_r <= r <= max_r:
            continue
        circularity = cv2.contourArea(c) / (np.pi * r * r + 1e-6)
        if circularity < min_circularity:
            continue
        # Ball-likeness: round, and close to the calibrated ball size.
        size_fit = 1.0
        if expected_r:
            size_fit = min(r, expected_r) / max(r, expected_r)
        candidates.append({"x": float(x), "y": float(y), "r": float(r),
                           "circularity": float(circularity),
                           "score": float(circularity * size_fit)})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def detect_with_diagnostics(frame, lower, upper, min_r, max_r, motion_mask=None,
                            min_circularity=MIN_CIRCULARITY):
    """Detect the ball and report why each rejected candidate failed.

    Returns (best, diag) where diag counts rejections by reason and records
    the near-misses, so a clip can be profiled for *why* detection drops.
    """
    mask = color_mask(frame, lower, upper, motion_mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diag = {"candidates": len(contours), "too_small": 0, "too_large": 0,
            "not_circular": 0, "best_rejected_r": None, "best_rejected_circ": None}
    best = None
    for c in contours:
        (x, y), r = cv2.minEnclosingCircle(c)
        circularity = cv2.contourArea(c) / (np.pi * r * r + 1e-6)
        if r < min_r:
            diag["too_small"] += 1
            if diag["best_rejected_r"] is None or r > diag["best_rejected_r"]:
                diag["best_rejected_r"] = round(float(r), 1)
            continue
        if r > max_r:
            diag["too_large"] += 1
            continue
        # Tolerate motion-blur elongation but reject long thin streaks (limbs, lines)
        if circularity < min_circularity:
            diag["not_circular"] += 1
            if diag["best_rejected_circ"] is None or circularity > diag["best_rejected_circ"]:
                diag["best_rejected_circ"] = round(float(circularity), 2)
            continue
        if best is None or r > best[2]:
            best = (x, y, r)
    diag["mask"] = mask
    return best, diag


def detect_ball_by_color(frame, lower, upper, min_r, max_r, motion_mask=None,
                         min_circularity=MIN_CIRCULARITY):
    """Largest plausibly-circular blob in the color mask, or None."""
    best, _ = detect_with_diagnostics(frame, lower, upper, min_r, max_r,
                                      motion_mask, min_circularity)
    return best
