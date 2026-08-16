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

### 3.1 Placement (end-on; documented user guidance)

**v0.4 reverses v0.3 on this point.** The previous version required a side-on
camera and called "behind the server" unusable. Field reality overruled it:
most gyms have no sideline space for a tripod, and users film from behind the
server. The verdict was true of the *method* v0.3 assumed (one fixed
metres-per-pixel scale applied to pixel displacement), not of the camera
position — so the method changed instead.

- **Angle:** behind the server, looking down the flight path. This is the
  worst possible geometry for measuring pixel displacement and the *best* for
  measuring apparent size, which is what §6.3 now uses. A ball flying away
  barely moves across the frame but shrinks steadily and measurably.
- **Position:** behind the endline, ideally raised enough that the server's
  body does not occlude the ball at contact. Tripod strongly preferred —
  static-candidate rejection (§6.1) assumes a roughly fixed camera.
- **Distance trade-off:** the ball must stay resolvable at the far end of its
  flight. On reference footage it spans 12–31 px across a rally; below roughly
  10 px the radius estimate degrades faster than the fit can absorb.

### 3.2 Depth is measured, not assumed

The v0.3 "measurement corridor" is **gone**. It existed because a single fixed
scale is only correct at one distance, so a ball flying a lane 1 m off the
calibrated plane read ~14% wrong at 8 m. Depth is now recovered per frame from
the ball's known 21 cm diameter (§6.3), so the ball may fly any lane.

What replaces it is a different error budget, driven by how precisely the
ball's apparent radius can be measured. Since `Z = f·D/w`, **fractional depth
error equals fractional radius error**. Measured on reference end-on footage:

| radius estimator | frame-to-frame noise | speed error at 12 m |
|---|---|---|
| **bounding-box max side** | **3.2%** | **±0.68 m/s (2.7%)** |
| segmentation mask area | 11.7% | ±2.51 m/s (10.0%) |
| mask minor axis | 20.3% | ±4.35 m/s (17.4%) |

The bounding box wins decisively, despite the mask being the more
sophisticated-looking option — this is why `analysis.ball_radius_px` uses the
box and the segmentation masks go unused.

Two properties keep this workable:

- **Constant bias cancels.** The box is not the ball's true silhouette, but
  calibration measures a reference ball through the *same* estimator, so any
  systematic scale factor divides out. Calibration and measurement must never
  use different estimators.
- **Noise averages down.** Per-frame depth error at 10 m is ~30 cm, comparable
  to how far the ball moves between frames at 30 fps — so frame-to-frame
  differencing is hopeless. Fitting the whole flight and taking the slope
  reduces the error by √N (§6.4).

### 3.3 Multi-hit / multi-ball behavior

One ball, one track (PRD non-goal). The detector selects the most prominent
valid blob; since closer = bigger, simultaneous balls resolve to the nearest
one and others are ignored — not measured wrong, just not measured. Serial
hits (a line of players serving one at a time) are the supported coaching
workflow.

## 4. Module Breakdown

| File | Responsibility | PRD refs |
|---|---|---|
| `webapp.py` | Flask upload UI, background job queue, CSV/video download | F6, F7 |
| `analysis.py` | End-to-end clip → `AnalysisResult`; the one path both front ends use | F1, F3, F6 |
| `track.py` | CLI: tracking, `--speed`, `--calibrate` | F4, F6 |
| `tracking/detectors.py` | Sliced YOLO inference + shape gate → candidates per frame | F1, F9 |
| `tracking/trackers.py` | Ball selection (static suppression), player ByteTrack | F2 |
| `tracking/annotators.py` | Ball trail and player overlays | F4 |
| `calibration.py` | Camera intrinsics; depth from apparent ball size; persists to config | F3, F5 |
| `speed.py` | 3D trajectory fit, flight segmentation, per-hit peak | F3, F4 |
| `config.json` | *(exists)* camera intrinsics, ball diameter | — |

Superseded from v0.3: `detector.py` (HSV) became `tracking/detectors.py` (YOLO),
so `hsv_calibrate.py` and the colour-calibration wizard step are gone —
F9 arrived early and F5's colour tuning became unnecessary. `setup_wizard.py`
collapses to a single `--calibrate` invocation now that there are no court
clicks to collect.

Flat module layout — the project is small enough that a `src/` tree would be
premature.

## 5. Data Model

```python
@dataclass
class BallSample:
    frame_idx: int
    timestamp: float        # seconds; file mode derives it from frame index
    center_px: tuple[float, float]
    radius_px: float        # the depth cue — see §3.2 on which estimator
    confidence: float

@dataclass(frozen=True)
class CameraIntrinsics:
    focal_px: float
    principal_x: float
    principal_y: float
    ball_diameter_m: float = 0.21
    # depth_m(radius_px) and to_world(center_px, radius_px) -> (X, Y, Z)

@dataclass
class Hit:
    start_time: float
    end_time: float
    peak_speed_kmh: float   # from the 3D trajectory fit (§6.4)
    samples: list[float]    # instantaneous speeds, for CSV/debugging
    n_points: int           # fit support
    mean_depth_m: float
    truncated: bool         # ball left frame / track lost mid-flight
    # .reliable == n_points >= 8 and not truncated
```

`Hit.reliable` exists because a quadratic through five points fits anything;
the peak it reports is then a property of the noise. A reading that cannot be
trusted must say so rather than sit in the table looking like the others.

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

### 6.3 Camera Calibration (F3, F5)

One measurement, once per device, never repeated:

```bash
track.py --source reference.mp4 --calibrate <radius_px> <distance_m>
```

Film the ball held at a measured distance, read off its apparent radius, and
`focal_length_from_reference` inverts `Z = f·D/(2r)` to get `focal_px`. The
principal point is assumed to be the frame centre — an offset of a few pixels
moves the lateral estimate by millimetres at these distances. Result is saved
to `config.json`.

No court markings are needed, which matters: the previous design required
clicking court lines, and half the point of the end-on view is that it works
in a gym where you cannot see or reach useful lines.

**The estimator must match.** Calibration and measurement both go through
`analysis.ball_radius_px`. The bounding box is not the ball's true silhouette,
so it carries a systematic bias — which cancels exactly, and only if both ends
measure the same way. Mixing estimators would put that bias straight into
every distance.

Fallback: `focal_length_from_fov` assumes a 70° horizontal field of view when
no calibration exists. Speeds then scale linearly with the true focal length,
so they are indicative and the UI labels them as such.

### 6.3a Ball Selection (F1)

Candidates come from sliced YOLO inference plus a shape gate (`--ball-max-size`,
`--ball-max-aspect`). Choosing among them uses **static suppression**: a
candidate reappearing at the same position *and the same apparent size* over
the last `static_window` frames is furniture, and the most confident survivor
wins.

Both halves of that test are load-bearing. Position alone was tried first and
discarded: filming from behind the server, a ball flying away holds nearly the
same image position for its whole flight, so a position-only rule threw away
86% of real detections on end-on footage. A net pole holds its size; a
receding ball does not.

This replaced Roboflow's centroid filter, which buffered every candidate and
so got anchored onto exactly the static objects it was meant to reject.

### 6.4 Speed Calculation (F3, F4)

Two outputs with different jobs:

- **Live readout** (overlay, latency-critical): finite difference between
  consecutive detections, median-of-3 smoothed. Jittery but immediate.
- **Peak speed per hit** (the headline number, PRD Q2): **3D trajectory fit.**
  Every sample is lifted to `(X, Y, Z)` metres via `intrinsics.to_world`, then
  a quadratic is fitted to each axis over the flight, differentiated, and the
  maximum magnitude taken. Raw per-frame speeds carry jitter, and
  max-of-noisy-samples is biased high — every peak would over-report by the
  luckiest upward jitter.

  Fitting is not merely a de-biasing nicety here, it is what makes the
  measurement possible at all. Per-frame depth error at 10 m is roughly 30 cm,
  comparable to how far the ball travels between frames at 30 fps, so a single
  frame pair carries almost no signal. Fitting ~15 of them cuts the error by
  √N to about ±0.7 m/s.

**Depth is the noisy axis, and it is quarantined.** Radius is median-smoothed
before becoming depth, and every *segmentation* decision — open, close, split —
runs on image-plane motion, which is precise. Only the reported speed uses
depth. A noisy Z must never be allowed to fabricate a "contact" that splits a
good flight in half.

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
- **Synthetic end-to-end validation (`tools/synthetic_validation.py`), the
  ground truth we have today:** a real volleyball, cut from real footage, is
  composited onto a real gym background along a trajectory we specify exactly,
  then the production pipeline runs over the result. It exercises detection,
  radius estimation, depth-from-size, the 3D fit and peak extraction against a
  number known to the millimetre.

  Current result: **72.3 km/h measured against 71.8 km/h true, +0.7%**, from a
  single 26-point segment with the ball detected in 30/30 frames.

  It does not reproduce motion blur or a real lens, so it validates the maths
  and the plumbing, not the optics. It is a substitute for the drop test, not
  a replacement for it.

- **Gravity drop tests (still required, needs a camera):** film a ball in free
  fall; physics fixes its true speed at every instant (v = g·t — after 0.5 s,
  exactly 17.6 km/h). Zero equipment, and unlike the synthetic clip it
  exercises the real optics. These clips become regression tests against the
  ±10% target.
- **Recorded-clip regression tests:** fixed set of serve/spike clips
  (phone slo-mo per PRD R2) with best-available reference speeds; pipeline
  output asserted within tolerance.
- **Field validation (M4, PRD §9):** side-by-side radar gun or
  known-distance timing, ≥ 8/10 within ±10% — final confirmation after the
  cheap physics-based validation already passes.
- Overlay rendering and live-camera wiring are verified by running the
  app, not unit tested.

## 9. Risks Carried Into Implementation

- **Detection recall is the binding constraint.** The ball is sometimes
  plainly visible and simply not detected, and nothing downstream recovers
  that. Fine-tuning on end-on footage — weighted toward far range, with hard
  negatives for net poles, banners and scoreboards — is the fix.
- **Radius noise sets the accuracy floor** (§3.2). Depth error equals radius
  error, so any change to the detector or the estimator changes the error
  budget and must be re-measured, not assumed.
- **Motion blur** inflates apparent radius exactly at peak speed, which biases
  depth *low* at the moment we care most. Elongation measured at a median 1.13
  on reference footage, so the effect is small there — but it grows with ball
  speed and is the first thing to check on genuinely fast serves.
- **Undersampling fast hits** (PRD R2): at 30 fps a half-second flight is only
  ~15 samples. 60–120 fps is the cheapest available improvement.
- **Static suppression assumes a fixed camera** (§6.3a). A hard pan makes
  static objects move, and they start surviving the filter.
- **False positives that move** — a thrown second ball, a light on a moving
  surface — defeat every filter here. Only a better detector helps.

## 10. Explicitly Deferred (matches PRD non-goals)

No player/pose tracking, no multi-ball, no spin/trajectory prediction, no
mobile port, no cloud/accounts. None get placeholder hooks; each would be a
new module behind existing interfaces, not a rewrite.
