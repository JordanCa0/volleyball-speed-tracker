# TDD: Volleyball Ball-Speed Tracker

**Status:** Draft v0.1 — for review
**Author:** Jordan (with Claude)
**Date:** 2026-07-10
**Companion to:** [PRD.md](./PRD.md)

## 1. Purpose

This document describes how we'll build the system described in the PRD:
module breakdown, algorithms, data flow, file layout, and how the four
milestones (M1–M4) map to concrete engineering tasks. It assumes the stack
already implied by the repo — Python + OpenCV, HSV color-threshold detection
(`hsv_calibrate.py`, `config.json` already exist) — rather than proposing a
new stack from scratch.

## 2. Architecture Overview

Single-process pipeline, one frame in → one (position, speed) reading out.
No microservices, no external processes — this is a desktop CV app.

```
                ┌─────────────┐
 camera/file →  │  Capture    │  frame (BGR), timestamp
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │  Detector   │  → Detection | None (center_px, radius_px, confidence)
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │  Tracker    │  → Track (position history, gap-tolerant)
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │ SpeedEngine │  → instantaneous speed, hit segmentation, peak
                └──────┬──────┘
                       ▼
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌─────────────┐           ┌─────────────┐
   │  Overlay/UI  │           │  SessionLog │ → CSV export
   └─────────────┘           └─────────────┘
```

Each stage is a plain Python class with a small, testable interface — no
stage reaches into another's internals. This matters because M1/M2 need to
run headless against recorded clips (no UI, no live camera) using the exact
same Detector/Tracker/SpeedEngine code that M3 wires up to a live window.

## 3. Module Breakdown

| File | Responsibility | PRD refs |
|---|---|---|
| `capture.py` | Wraps `cv2.VideoCapture` for both camera index and file path; normalizes fps/frame timestamps | F6 |
| `detector.py` | HSV threshold + contour → `Detection` per frame | F1, F9 (future) |
| `tracker.py` | Frame-to-frame association, occlusion tolerance, motion trail | F2 |
| `calibration.py` | Scale calibration (ball-diameter or two-point) + persists to config | F3, F5, F8 (future) |
| `hsv_calibrate.py` | *(exists)* interactive color calibration tool | F5 |
| `speed.py` | px/frame → km/h + mph, hit segmentation, peak tracking | F3, F4 |
| `overlay.py` | Draws detection, trail, speed readout on frame | F4 |
| `session_log.py` | In-memory hit log + CSV export | F7 |
| `main.py` | Wires the pipeline together for both live and file-replay modes | F4, F6 |
| `config.json` | *(exists)* camera index, HSV bounds, ball diameter, min radius | — |

This is a flat module layout (no package nesting) — the project is small
enough that a `src/` tree or plugin architecture would be premature.

## 4. Data Model

```python
@dataclass
class Detection:
    frame_idx: int
    timestamp: float        # seconds, from capture clock
    center_px: tuple[float, float]
    radius_px: float

@dataclass
class TrackPoint:
    detection: Detection | None   # None = gap frame (occlusion/miss)

@dataclass
class Hit:
    start_time: float
    end_time: float
    peak_speed_kmh: float
    samples: list[float]    # instantaneous speeds, for CSV/debugging

@dataclass
class Track:
    points: deque[TrackPoint]     # bounded, e.g. last 60 frames for trail rendering
```

## 5. Algorithms

### 5.1 Detection (F1)

v1 reuses the approach already scaffolded in `hsv_calibrate.py`:
1. Convert frame to HSV.
2. `cv2.inRange` with calibrated `hsv_lower`/`hsv_upper`.
3. Morphological erode+dilate to remove speckle noise.
4. `cv2.findContours` → largest contour above `min_radius_px`.
5. `cv2.minEnclosingCircle` → `(center_px, radius_px)`.

This is deliberately simple and fast (sub-millisecond per frame at 720p),
which matters for the ≥30fps target. F9 (YOLO) is a drop-in replacement
behind the same `Detector` interface later — not a v1 concern.

### 5.2 Tracking (F2)

Ball motion between consecutive frames at 30fps is small and roughly
linear, so we don't need a full Kalman filter for v1:

- Maintain last known position + velocity estimate (simple finite
  difference over the last 2–3 good detections).
- On a miss (`Detector` returns `None`), predict the next position by
  extrapolating velocity and widen the acceptance radius for re-acquiring
  the ball.
- Tolerate up to 5 consecutive gap frames (PRD F2). Beyond that, drop the
  track — a hit ends, or a new track starts on next detection.
- A candidate detection is accepted into the current track only if it's
  within a plausible distance of the predicted position (rejects
  spurious same-color blobs elsewhere in frame, e.g. skin tone or gym
  equipment).

This is the highest-risk module for false tracks in a "messy background"
gym — see §8 risks.

### 5.3 Scale Calibration (F3)

Two methods, both behind the same `calibration.py` interface:

- **Ball-diameter method (v1 default):** ball's real diameter (0.21 m,
  already in `config.json`) divided by its detected `radius_px * 2` on
  each frame gives a per-frame meters-per-pixel scale. This adapts
  somewhat to the ball moving toward/away from the camera, but the PRD
  already documents that depth-axis motion is unreliable (§7 known
  limitations) — this method only corrects for it partially.
- **Two-point method:** user clicks two points of known real-world
  distance (e.g. court line markers) at setup; gives a fixed
  meters-per-pixel scale for the whole session. Simpler, but wrong if the
  ball moves significantly closer/farther from the camera than the
  calibration plane.
- **Homography (F8, later):** maps full court-line geometry to a ground
  plane for perspective-correct scale everywhere in frame. Deferred to M4.

### 5.4 Speed Calculation (F3, F4)

For each pair of consecutive good detections:

```
speed_px_per_s = |Δposition_px| / Δtimestamp
speed_m_per_s  = speed_px_per_s * meters_per_pixel   (from calibration.py)
speed_kmh      = speed_m_per_s * 3.6
```

Use actual frame timestamps (not assumed fixed fps) since dropped frames
or variable-fps camera input would otherwise silently corrupt the speed
calc — this is a common bug source in naive implementations.

Apply light smoothing (e.g. median-of-3 over instantaneous speed samples)
to suppress single-frame detection jitter before it hits peak-speed
tracking, without adding meaningful latency (PRD requires <1s readout).

### 5.5 Hit Segmentation (Q3 in PRD, proposal accepted)

- A **hit** begins when instantaneous speed crosses above a threshold
  (config value, tunable — start around 15–20 km/h to exclude a ball
  being carried/placed).
- Peak speed is tracked as the max instantaneous speed while the hit is
  "open."
- The hit **ends** when speed drops back below the threshold, or the
  track is lost (ball exits frame / occlusion exceeds tolerance).
- On end, emit a `Hit` record to `SessionLog`.

### 5.6 Frame Rate / Latency Budget

Target ≥30fps sustained (33ms/frame budget). Rough allocation:
- Capture: <1ms (buffered read)
- HSV detect: ~2–5ms at 720p
- Tracking/speed math: <1ms
- Overlay draw + `cv2.imshow`: ~5–10ms

This leaves headroom; if 60–120fps input is used (PRD stretch), we may
need to move capture to its own thread with a small ring buffer so
`imshow` doesn't stall frame acquisition — flagged as a task for M3, not
needed for M1/M2 (which run headless against files).

## 6. Milestone → Technical Task Mapping

| Milestone | Tasks |
|---|---|
| **M1** — Detection on recorded clips | `capture.py` (file mode), `detector.py`, `tracker.py`, wire into a headless CLI runner that overlays+saves annotated video for visual QA; finalize `hsv_calibrate.py` workflow |
| **M2** — Speed on recorded clips | `calibration.py` (ball-diameter method), `speed.py`, a small ground-truth test harness (§7) comparing computed peak speed to manually-measured reference clips |
| **M3** — Live mode | `capture.py` (camera mode), `overlay.py` live window, `session_log.py` + CSV export, main.py live entrypoint |
| **M4** — Accuracy pass | Homography calibration (F8), variable/high-fps handling, field validation against radar gun or known-distance timing |

## 7. Testing Strategy

- **Unit tests** for pure-logic modules (`speed.py` math, `tracker.py` gap
  handling, hit segmentation state machine) using synthetic position
  sequences — no camera or video files needed, fast to run in CI.
- **Recorded-clip regression tests** (M1/M2): a small fixed set of test
  clips (phone slo-mo recommended per PRD R2) with hand-measured ground
  truth (known throw distance/time, or side-by-side radar reading).
  Running the pipeline against these clips should stay within the ±10%
  accuracy target — this becomes an automated check, not just manual eyeballing.
- **Manual/field validation** (M4, per PRD §9): side-by-side radar gun or
  known-distance timing trials, ≥8/10 within target.
- No formal test coverage requirement beyond "detection/tracking/speed
  math are covered" — overlay rendering and live-camera wiring are best
  verified by running the app (per project convention), not unit tested.

## 8. Risks Carried Into Implementation

These map directly to PRD §10 but called out here because they affect
module design specifically:

- **False positive tracks** (R1 — background clutter matching ball color):
  addressed by the tracker's predicted-position gating (§5.2), not just
  the detector. If this proves insufficient in testing, F9 (YOLO) is the
  planned escape hatch — `Detector` is already an isolated interface for
  that swap.
- **Undersampling fast hits** (R2): speed math already uses real
  timestamps (§5.4) so it degrades gracefully with variable fps; the open
  question is whether 30fps detection is reliable at all on spikes
  >100km/h — this needs empirical testing early (M2), not just at M4.
- **Depth-axis motion**: explicitly out of scope for accuracy in v1
  (PRD §7); calibration.py's ball-diameter method is a partial mitigation,
  documented as such rather than treated as a fix.

## 9. Explicitly Deferred (matches PRD non-goals)

No player/pose tracking, no multi-ball, no spin/trajectory, no mobile
port, no cloud/accounts — none of these get placeholder hooks in the
architecture; adding them later is a new module behind the existing
interfaces, not a rewrite.
