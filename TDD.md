# TDD: Volleyball Ball-Speed Tracker

**Status:** Draft v0.3 — for review
**Author:** Jordan (with Claude)
**Date:** 2026-07-10
**Companion to:** [PRD.md](./PRD.md)

**Changes in v0.3:** hit segmentation reframed as flight segments split at
velocity discontinuities (handles blocks/deflections, §6.5); measurement
window and action-classification limits made explicit (§6.5).
**Changes in v0.2:** camera geometry section (§3); guided setup wizard with
court-line calibration (§6.3); scale calibration no longer per-frame
ball-diameter by default; trajectory-fit speed instead of raw
finite-difference peak; max-plausible-speed tracker gating; threaded capture
promoted to a firm M3 task; AE/AWB lock + background-subtraction option;
gravity drop-test validation.

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
 camera/file →  │  Capture    │  frame (BGR), capture-time timestamp
                └──────┬──────┘   (threaded grab, 1-frame buffer — §6.6)
                       ▼
                ┌─────────────┐
                │  Detector   │  → Detection | None (center_px, radius_px)
                └──────┬──────┘   (HSV mask ∧ motion mask — §6.1)
                       ▼
                ┌─────────────┐
                │  Tracker    │  → Track (position history, gap-tolerant)
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │ SpeedEngine │  → live speed, hit segmentation,
                └──────┬──────┘    trajectory-fit peak speed
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

## 3. Camera Geometry & Measurement Volume

These constraints are physics, not implementation choices — they bound what
any single-camera design can measure and drive the setup wizard (§6.3).

### 3.1 Placement (documented user guidance + wizard defaults)

- **Angle:** lens perpendicular to the flight path (side-on). A 2D camera
  only measures the velocity component parallel to its image plane;
  measured speed = true speed × cos(θ), where θ is the angle between the
  flight direction and the image plane. Within ±15° of perpendicular the
  cosine error stays under ~3.5%; at 30° it's already 13% — over the whole
  ±10% accuracy budget. Camera behind the server is unusable.
- **Position:** on the sideline, roughly level with the mid-height of the
  flight path (tripod ~1.5–2.5 m), 6–10 m back — far enough to frame the
  whole flight and flatten perspective, close enough that the ball stays
  ≥ `min_radius_px` for detection.
- **Distance trade-off:** farther camera → smaller depth error (§3.2) and
  full flight in frame, but fewer pixels on the ball. Wizard should preview
  detected ball radius so the user can verify it's above threshold.

### 3.2 Measurement corridor (single-camera depth limitation)

The px→m scale is calibrated at one distance from the camera. A ball flying
a lane Δd closer/farther than the calibration plane reads fast/slow by
roughly D/(D∓Δd) (camera distance D). At D = 8 m, a 1 m lane drift is ~14%
error; at D = 12 m, ~9%. Consequences, documented for v1:

- Accuracy holds in a **corridor around the calibrated lane** (~±0.5–1 m
  depending on D), not across the whole court. Users should have players
  hit from roughly the same lane — consistent with radar-gun workflows.
- Homography (F8, M4) removes this for position *on the court plane* by
  computing per-position scale; it still does not recover true depth-axis
  velocity (PRD §7 known limitation stands).

### 3.3 Multi-hit / multi-ball behavior

One ball, one track (PRD non-goal). The detector selects the most prominent
valid blob; since closer = bigger, simultaneous balls resolve to the nearest
one and others are ignored — not measured wrong, just not measured. Serial
hits (a line of players serving one at a time) are the supported coaching
workflow.

## 4. Module Breakdown

| File | Responsibility | PRD refs |
|---|---|---|
| `capture.py` | Threaded frame grab (camera or file), capture-time stamping, 1-frame buffer | F6 |
| `detector.py` | HSV mask ∧ background-subtraction mask → `Detection` per frame | F1, F9 (future) |
| `tracker.py` | Frame-to-frame association, occlusion tolerance, motion trail | F2 |
| `calibration.py` | Court-line two-point scale (default), ball-diameter fallback, homography (F8 later); persists to config | F3, F5, F8 |
| `setup_wizard.py` | Guided setup flow: framing preview → court-line clicks → color calibration → AE/AWB lock → save | F5 |
| `hsv_calibrate.py` | *(exists)* interactive HSV tuner, invoked as the color step of the wizard | F5 |
| `speed.py` | Live finite-difference readout + trajectory-fit peak, hit segmentation | F3, F4 |
| `overlay.py` | Draws detection, trail, speed readout on frame | F4 |
| `session_log.py` | In-memory hit log + CSV export | F7 |
| `main.py` | Entrypoints: `--setup` (wizard), live mode, `--source <file>` replay | F4, F6 |
| `config.json` | *(exists)* camera index, HSV bounds, scale calibration, ball diameter, min radius | — |

Flat module layout — the project is small enough that a `src/` tree would be
premature.

## 5. Data Model

```python
@dataclass
class Detection:
    frame_idx: int
    timestamp: float        # seconds, monotonic clock at grab time
    center_px: tuple[float, float]
    radius_px: float

@dataclass
class TrackPoint:
    detection: Detection | None   # None = gap frame (occlusion/miss)

@dataclass
class Hit:
    start_time: float
    end_time: float
    peak_speed_kmh: float   # from trajectory fit (§6.4)
    samples: list[float]    # instantaneous speeds, for CSV/debugging

@dataclass
class ScaleCalibration:
    method: Literal["court_line", "ball_diameter"]
    meters_per_px: float           # court_line: fixed for session
    reference_points_px: list[tuple[float, float]]  # reusable for F8 homography
```

## 6. Algorithms

### 6.1 Detection (F1)

Base pipeline as scaffolded in `hsv_calibrate.py`:
1. Convert frame to HSV; `cv2.inRange` with calibrated bounds.
2. Morphological erode+dilate to remove speckle.
3. `cv2.findContours` → best contour above `min_radius_px`.
4. `cv2.minEnclosingCircle` → `(center_px, radius_px)`.

Two robustness additions over v0.1:

- **AND with a motion mask.** Run `cv2.createBackgroundSubtractorMOG2` and
  require pixels to be both ball-colored *and* moving. Rejects static
  same-hue clutter (floors, walls, posters) and slow-moving skin tones at
  ~1–2 ms/frame cost — a large share of the robustness F9 (YOLO) was
  reserved for, nearly free. Config flag to disable for A/B testing.
- **Lock auto-exposure / auto-white-balance** after color calibration
  (wizard step, §6.3). Camera auto-adjustment shifts the ball's rendered
  hue out of the calibrated band mid-session — the most likely cause of
  "worked at setup, fails 20 minutes in" (PRD 30-min stability metric).
  Where the backend can't lock AE/AWB, wizard warns and suggests stable
  lighting.

Known limitation: a net-occluded ball can split into two contours;
largest-contour selection then jumps. If this shows up in M1 footage,
mitigate by merging contours within a ball-diameter of each other before
circle fitting.

F9 (YOLO) remains a drop-in `Detector` replacement later — not v1.

### 6.2 Tracking (F2)

- Maintain last known position + velocity estimate (finite difference over
  the last 2–3 good detections).
- On a miss, extrapolate position and widen the acceptance radius; tolerate
  ≤ 5 consecutive gap frames (PRD F2), then drop the track.
- **Gating: bound by maximum plausible ball speed, not predicted position.**
  Prediction-based gating fails exactly at contact — a hit is a velocity
  discontinuity the extrapolation can't foresee, so the first post-contact
  detection would be rejected and every hit's start truncated. Instead,
  accept any detection within `max_speed_kmh` (config, default 130)
  converted to px/frame via current scale. Distant same-color blobs still
  fail this test; legal hits always pass.

### 6.3 Setup Wizard & Scale Calibration (F3, F5)

`main.py --setup` runs a guided flow (per PRD's 2-minute setup goal):

1. **Framing preview:** live feed with guidance overlay ("place camera
   side-on to the flight path"; §3.1 text). Shows detected-ball radius so
   the user can confirm the ball is large enough at this distance.
2. **Court-line scale calibration (v1 default):** user clicks two points a
   known real distance apart. Court geometry is standardized, so the wizard
   offers presets: attack line → center line along the sideline (exactly
   3.00 m), half-court sideline (9.00 m), net height (2.43 m / 2.24 m) for
   a vertical reference. Custom distance also allowed (any marked span).
   Clicked points are persisted; collecting ≥ 4 court points here later
   feeds F8 homography directly with no new UX.
3. **Color calibration:** existing `hsv_calibrate.py` flow.
4. **Lock AE/AWB** (§6.1) and save everything to `config.json`.

Scale methods, in order of preference:

- **Court-line two-point (default):** fixed meters-per-pixel from large,
  stationary references. Accurate within the measurement corridor (§3.2).
- **Ball-diameter (fallback,** e.g. no visible lines — backyard/beach):
  demoted from v0.1's per-frame default. Per-frame `radius_px` is noisy
  (±1–2 px on a 10–25 px radius is a 5–15% scale error) and motion blur
  elongates the ball exactly at peak speed, inflating apparent radius. If
  used, take the **median radius over the slow early portion of the
  track** (or a stationary pre-hit frame) and hold it fixed per hit —
  never per-frame at speed.
- **Homography (F8, M4):** per-position scale from ≥ 4 court points;
  removes the corridor restriction for on-court-plane motion.

### 6.4 Speed Calculation (F3, F4)

Two outputs with different jobs:

- **Live readout** (overlay, latency-critical): finite difference between
  consecutive detections, median-of-3 smoothed. Jittery but immediate.
- **Peak speed per hit** (the headline number, PRD Q2): **trajectory fit.**
  Raw per-frame speeds carry pixel-jitter noise, and max-of-noisy-samples
  is biased high — every peak would over-report by the luckiest upward
  jitter. Instead, fit a low-order polynomial to x(t), y(t) over the hit
  window (in-flight paths are near-ballistic, so quadratic fits well),
  differentiate the fit, and take its maximum. Fit incrementally as samples
  arrive so the readout still lands within the PRD's 1-second latency;
  finalize on hit end. ~10 lines of `numpy.polyfit`.

Both use **actual grab timestamps** (monotonic clock stamped in the capture
thread, §6.6), never assumed-fixed fps — buffered or dropped frames
otherwise compress Δt and silently inflate speeds.

### 6.5 Hit Segmentation: Flight Segments (PRD Q3/Q4, refined)

The primitive the system can actually measure is not "a hit" but an
**uninterrupted ballistic flight segment** — a stretch of track where the
ball moves under gravity alone. Every contact (hand, block, forearms,
floor, net) breaks the arc, and a broken arc is detectable from kinematics.

- A segment **opens** when live speed crosses a threshold (config,
  ~15–20 km/h — excludes carried/placed balls).
- A segment **splits** on a **velocity discontinuity**: direction change
  beyond a threshold (~30–45° between consecutive velocity vectors) or a
  speed jump gravity can't produce. This is how deflections are handled —
  spike → block → floor logs as three segments, each with its own peak,
  instead of one merged record or arbitrary occlusion-driven fragments.
  Peak speed from the attack segment is the attack speed; the post-block
  segment's peak is reported as its own entry.
- A segment **closes** when speed drops below threshold, the track is lost
  (> 5 gap frames), or a split occurs.
- On close, finalize the trajectory fit (§6.4 — fits are only valid within
  one ballistic arc, so segment and fit boundaries coincide by design) and
  emit a `Hit` to `SessionLog`.

**Measurement window semantics:** the headline number is the segment's
peak speed, which physically occurs just after contact — drag only slows
the ball from there (a hard serve sheds 10–20% crossing the court). This
matches radar-gun convention (PRD Q2). The window is *not* contact-to-
ground and the number is not a flight average.

**Action classification — none in v1 (PRD Q4 accepted).** Distinguishing
serve/spike/set/pass requires player context (pose, court position),
which is an explicit v1 non-goal; ball kinematics alone can't do it
reliably. The speed threshold acts as a crude filter — sets and easy
passes (< ~25–30 km/h) fall below a ~30–40 km/h reporting threshold,
attacks fall above — with known overlap (hard-driven passes log as hits).
Intended use is drills, where the user supplies context ("everything fast
right now is a serve"); in open game play the log degrades to "all fast
flight segments regardless of cause," documented as a limitation.

Known edge case: a serve toss can exceed the open threshold, merging
toss+hit — but the toss→hit contact is itself a velocity discontinuity,
so the split logic separates them naturally; gets a regression test clip.

### 6.6 Capture & Timing (promoted from "maybe" to firm M3 task)

`cv2.VideoCapture` returns dequeue time, not capture time, and buffers
frames; under load, queued frames arrive in a burst, Δt collapses, and
speeds read absurdly high. This is an accuracy requirement, not a
throughput nicety:

- Dedicated capture thread grabs frames as they're ready and stamps
  `time.monotonic()` at grab; pipeline consumes the latest frame.
- `CAP_PROP_BUFFERSIZE = 1` where the backend honors it.
- File mode instead derives timestamps from the container's fps/PTS
  (exact by definition), keeping M1/M2 replay deterministic.

Frame budget at ≥ 30 fps (33 ms): detect ~2–5 ms, MOG2 ~1–2 ms,
track/speed < 1 ms, overlay+imshow ~5–10 ms — headroom remains for
60–120 fps input at reduced resolution.

## 7. Milestone → Technical Task Mapping

| Milestone | Tasks |
|---|---|
| **M1** — Detection on recorded clips | `capture.py` (file mode), `detector.py` (incl. MOG2 ∧ color), `tracker.py` (max-speed gating), headless CLI runner saving annotated video for visual QA |
| **M2** — Speed on recorded clips | `calibration.py` (two-point + ball-diameter fallback), `speed.py` (finite-diff live + trajectory fit), **gravity drop-test validation** (§8), ground-truth clip harness; empirical check: is 30 fps detection viable on >100 km/h spikes? (PRD R2 — answer this here, not at M4) |
| **M3** — Live mode | `capture.py` camera mode with capture thread + monotonic stamping, `setup_wizard.py`, AE/AWB lock, `overlay.py`, `session_log.py` + CSV |
| **M4** — Accuracy pass | Homography from wizard-collected court points (F8), high-fps input handling, field validation vs. radar/known-distance (PRD §9) |

## 8. Testing Strategy

- **Unit tests** for pure logic (`speed.py` fit + conversion math,
  `tracker.py` gap handling and gating, hit-segmentation state machine)
  using synthetic position sequences — no camera/video needed, CI-fast.
- **Gravity drop tests (M2, primary ground truth):** film a ball in free
  fall; physics fixes its true speed at every instant (v = g·t — after
  0.5 s, exactly 17.6 km/h). Zero equipment, validates the entire chain
  (detection → scale → timestamps → math) end-to-end. These clips become
  automated regression tests against the ±10% target.
- **Recorded-clip regression tests:** fixed set of serve/spike clips
  (phone slo-mo per PRD R2) with best-available reference speeds; pipeline
  output asserted within tolerance.
- **Field validation (M4, PRD §9):** side-by-side radar gun or
  known-distance timing, ≥ 8/10 within ±10% — final confirmation after the
  cheap physics-based validation already passes.
- Overlay rendering and live-camera wiring are verified by running the
  app, not unit tested.

## 9. Risks Carried Into Implementation

- **False positive tracks** (PRD R1): now double-mitigated — motion∧color
  detection (§6.1) and max-speed gating (§6.2). YOLO (F9) remains the
  escape hatch behind the `Detector` interface if M1 footage still shows
  false tracks.
- **Undersampling fast hits** (PRD R2): timestamp-true speed math degrades
  gracefully, but whether 30 fps detection survives >100 km/h motion blur
  is answered empirically in M2 with slo-mo-vs-30fps comparisons of the
  same hits.
- **Measurement corridor** (§3.2): a *usage* constraint, not a bug —
  documented in setup guidance; homography (M4) relaxes it.
- **Depth-axis motion**: out of scope for v1 accuracy (PRD §7); no
  single-camera fix planned.

## 10. Explicitly Deferred (matches PRD non-goals)

No player/pose tracking, no multi-ball, no spin/trajectory prediction, no
mobile port, no cloud/accounts. None get placeholder hooks; each would be a
new module behind existing interfaces, not a rewrite.
