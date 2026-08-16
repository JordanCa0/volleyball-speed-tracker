# PRD: Volleyball Ball-Speed Tracker

**Status:** Draft v0.1 — for review
**Author:** Jordan (with Claude)
**Date:** 2026-07-07

## 1. Problem Statement

Players and coaches have no accessible way to measure how fast a volleyball is
being hit (serves, spikes). Commercial systems (Hawk-Eye, radar guns) are
expensive, require dedicated hardware, or aren't designed for volleyball.
A camera-only tool would make speed feedback available to anyone with a phone
or laptop.

## 2. Goals

- Measure the speed of a volleyball in flight (serves and spikes) from live
  camera footage, with no extra hardware beyond a camera.
- Display speed in real time (or near-real time) while recording.
- Be simple enough for a single person to set up courtside in under 2 minutes.

## 3. Non-Goals (v1)

- Player tracking, pose estimation, or touch/contact detection.
- Multi-ball or multi-court scenarios.
- Spin rate or trajectory prediction.
- Native mobile app (deferred — see §8; v1 runs on a laptop).
- Match statistics, cloud storage, or accounts.

## 4. Target Users

| User | Need |
|---|---|
| Player training solo | Instant feedback on serve speed to track improvement |
| Coach running practice | Quick readings across multiple players' attempts |
| Recreational player | Fun/competitive "radar gun" experience |

## 5. User Experience (v1 prototype)

The camera sits **behind the server**, because that is where gyms have space.
Everything below follows from that (see TDD §3.1).

1. User films a rally from behind the endline — phone on a tripod, whole
   flight in frame.
2. One-time camera calibration (~1 minute, per device, never repeated):
   film a ball at a measured distance, and the app fits the camera's focal
   length. No court markings needed, no colour tuning.
3. User uploads the clip through the web UI (or runs the CLI) and gets back:
   - Peak speed per flight segment, in km/h and mph.
   - A quality flag per reading, so a truncated or too-short flight is not
     presented as though it were a clean measurement.
   - An annotated video showing what the detector locked onto.
4. Results export to CSV.

**Not live.** Analysis runs at roughly 1.4 fps, so a 60-second clip takes
about 20 minutes. Real-time overlay moves to a later milestone; the
upload-and-wait flow is what v1 delivers.

## 6. Functional Requirements

| ID | Requirement | Priority | Done |
|---|---|---|---|
| F1 | Detect the ball in each frame | P0 | ✅ |
| F2 | Select the real ball among candidates, rejecting static false positives | P0 | ✅ |
| F3 | Recover depth per frame from the ball's known diameter and convert to real-world speed | P0 | ✅ |
| F4 | Report peak speed per flight segment with a reliability flag | P0 | ✅ |
| F5 | One-time camera calibration that persists | P0 | ✅ |
| F6 | Work on recorded video files | P1 | ✅ |
| F7 | Hit log with timestamps, exportable to CSV | P1 | ✅ |
| F8 | Web UI for uploading clips and reading results | P1 | ✅ |
| F9 | Trained ML detector rather than colour thresholding | P2 | ✅ |
| F10 | Fine-tuned detector for end-on club footage | P1 | ☐ |
| F11 | Validation against ground truth (drop test, radar) | P0 | ☐ |
| F12 | Live real-time overlay | P2 | ☐ |

Dropped from v0.1: interactive colour calibration and court-line homography.
F9 arrived early, which made colour tuning unnecessary, and measuring depth
directly made homography pointless.

## 7. Accuracy & Performance Targets

- **Speed accuracy:** within ±10% of true speed for serves in the
  40–100 km/h range, with the camera behind the server.
  (Stretch: ±5% at 60 fps or higher.)
- **Latency:** speed readout visible within 1 second of the hit.
- **Frame rate:** pipeline sustains ≥ 30 fps on a modern laptop at 720p;
  supports 60–120 fps input for reduced motion blur where hardware allows.
- **Detection rate:** ball detected in ≥ 80% of in-flight frames under
  good lighting with a color-contrasted ball.

### Known accuracy limitations (accepted for v1)

- **Depth from ball size is the method now, not a limitation** — but it is the
  dominant error term. Fractional depth error equals fractional radius error;
  measured at 3.2% frame-to-frame, giving ~±0.7 m/s over a half-second flight
  (TDD §3.2). It relies on the ball being a standard 21 cm.
- **A single noisy frame cannot be trusted.** The number is only meaningful
  from a fitted flight; readings from fewer than 8 points, or from a flight
  truncated by the ball leaving frame, are flagged rather than reported as
  equals.
- Motion blur at 30 fps will degrade detection of very fast spikes
  (> 100 km/h); higher-fps input is the mitigation, not algorithm changes.
- **Detection recall is the current ceiling.** The ball is sometimes plainly
  visible and simply not detected; no amount of downstream maths recovers it.

## 8. Platform & Deployment

- **v1 (this PRD):** Python desktop app (macOS first) using a built-in or
  USB webcam, or a phone used as a webcam (e.g. Continuity Camera / IP
  camera stream). Development and tuning also runs against recorded clips.
- **v2 (future, out of scope here):** phone-native experience. Candidate
  paths: phone-streams-to-laptop companion mode, Android-native via
  Chaquopy, or a port of the pipeline to a mobile CV stack. Decision
  deferred until v1 validates the pipeline.

## 9. Success Metrics (prototype)

- A serve recorded side-by-side with a radar gun (or manually timed over a
  known distance) reads within the ±10% target in ≥ 8 of 10 trials.
- Setup-to-first-reading time under 2 minutes after initial calibration.
- Tracker runs a full 30-minute practice session without crashing or
  drifting into false readings.

## 10. Risks & Open Questions

| # | Risk / Question | Notes |
|---|---|---|
| R1 | ~~Ball colour vs. background~~ | Resolved: F9 replaced colour thresholding entirely |
| R2 | 30 fps may undersample fast hits | Open, and now quantified: ~15 samples per flight. Test 60/120 fps |
| R3 | **Detection recall is the ceiling** | The ball is sometimes visible and undetected; F10 is the fix |
| R4 | **No ground truth yet** | Speeds are self-consistent but unvalidated. M3 blocks any accuracy claim |
| R5 | Motion blur inflates apparent radius at peak speed, biasing depth low | Measured mild (1.13 elongation) on reference footage; recheck on fast serves |
| Q1 | Indoor, outdoor (grass/beach), or both for v1? | Indoor first — that is what the footage is |
| Q2 | Is "peak speed per hit" the right headline metric? | Settled: peak, matching radar-gun convention |
| Q3 | How is a "hit" segmented? | Settled: ballistic flight segment, split on velocity discontinuity (TDD §6.5) |
| Q4 | Do we need to distinguish serves from spikes/passes in v1? | Settled: no — everything above threshold is a "hit" |

## 11. Milestones

1. ~~**M1 — Detection on recorded clips**~~ *(done)*: pretrained volleyball
   detector, shape gating, static suppression (F1, F2, F9).
2. ~~**M2 — Speed on recorded clips**~~ *(done)*: depth from apparent size,
   3D trajectory fit, per-flight peak, calibration, web UI and CSV
   (F3, F4, F5, F6, F7, F8).
3. **M3 — Validation** *(next)*: gravity drop test and known-distance roll
   test against the ±10% target; then a radar or timed-distance field check
   (F11). **No speed number should be trusted until this is done** — the
   pipeline is self-consistent but has never been checked against a known
   truth.
4. **M4 — Recall**: fine-tune the detector on end-on footage weighted toward
   far range, with hard negatives (F10).
5. **M5 — Live mode**: real-time overlay, which needs a large speed-up first
   (F12).
