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

1. User places a laptop (or phone streaming to laptop) courtside with the
   camera viewing the flight path, roughly perpendicular to it.
2. User runs a one-time calibration step (~1 minute):
   - Color calibration: tune detection so the ball is reliably isolated.
   - Scale calibration: confirm ball diameter (standard 21 cm) or mark two
     points of known distance in frame.
3. User starts tracking mode. A live window shows:
   - The camera feed with the detected ball highlighted and a motion trail.
   - Current speed of the ball in flight (km/h and mph).
   - Peak speed of the most recent hit ("last hit: 74 km/h").
4. Each detected hit is logged with a timestamp and peak speed; the session
   log can be exported (CSV).

## 6. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| F1 | Detect the ball in each frame of a live camera feed | P0 |
| F2 | Track ball position across frames, tolerating brief occlusion/blur (≤ 5 frames) | P0 |
| F3 | Convert pixel displacement to real-world speed using known ball diameter | P0 |
| F4 | Display live speed overlay + peak speed per hit | P0 |
| F5 | Interactive color-calibration tool that persists settings | P0 |
| F6 | Work on recorded video files as well as live camera (for testing/tuning) | P1 |
| F7 | Session log of hits with timestamps, exportable to CSV | P1 |
| F8 | Court-line homography calibration as a more accurate alternative to ball-diameter scaling | P2 |
| F9 | Swap color detection for a trained ML detector (YOLO) for robustness in messy backgrounds | P2 |

## 7. Accuracy & Performance Targets

- **Speed accuracy:** within ±10% of true speed for serves in the
  40–100 km/h range, with the camera side-on to the flight path.
  (Stretch: ±5% with homography calibration.)
- **Latency:** speed readout visible within 1 second of the hit.
- **Frame rate:** pipeline sustains ≥ 30 fps on a modern laptop at 720p;
  supports 60–120 fps input for reduced motion blur where hardware allows.
- **Detection rate:** ball detected in ≥ 80% of in-flight frames under
  good lighting with a color-contrasted ball.

### Known accuracy limitations (accepted for v1)

- Single-camera depth estimation from ball size is noisy; speed along the
  camera axis (ball flying toward/away from camera) will be unreliable.
  v1 assumes a mostly side-on camera angle and documents this.
- Motion blur at 30 fps will degrade detection of very fast spikes
  (> 100 km/h); higher-fps input is the mitigation, not algorithm changes.

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
| R1 | Ball color vs. background — color-based detection fails on white balls against bright gym walls | Mitigation: start with a colored ball (blue/yellow); F9 (YOLO) is the long-term fix |
| R2 | 30 fps may undersample fast hits | Test with phone slo-mo (120/240 fps) clips early |
| Q1 | Indoor, outdoor (grass/beach), or both for v1? | Affects lighting assumptions and detection tuning |
| Q2 | Is "peak speed per hit" the right headline metric, or average speed over flight? | Radar guns report peak; proposal: peak |
| Q3 | How is a "hit" segmented? | Proposal: a hit starts when ball speed jumps above a threshold and ends when the ball is lost or slows below it |
| Q4 | Do we need to distinguish serves from spikes/passes in v1? | Proposal: no — everything above the speed threshold is logged as a "hit" |

## 11. Milestones

1. **M1 — Detection on recorded clips:** ball reliably detected and tracked
   in test footage (F1, F2, F5, F6).
2. **M2 — Speed on recorded clips:** pixel→real-world conversion and per-hit
   peak speed on known test clips (F3), validated against a ground truth.
3. **M3 — Live mode:** real-time overlay from webcam with session log
   (F4, F7).
4. **M4 — Accuracy pass:** homography option, higher-fps input handling,
   field validation against radar/known-distance timing (F8).
