# Volleyball Speed Tracker

Measure how fast a volleyball is hit (serves, spikes) using nothing but a
camera — no radar gun, no special hardware. See [PRD.md](PRD.md) for the
product vision and [TDD.md](TDD.md) for the technical design.

**Film from behind the server.** That is where gyms have space, so it is what
the system is built for.

## How it works

```
frame ─ 2x2 sliced inference ─ shape gate ─ static suppression
      └─ depth from apparent size ─ 3D trajectory fit ─ peak speed
```

A ball flying away from the camera barely moves across the frame — but it
shrinks, steadily and measurably. Since a volleyball is a standard 21 cm, its
apparent size *is* a distance measurement:

```
Z = focal_px × 0.21 / diameter_px          depth in metres, every frame
X = (u − cx) × Z / focal_px                lateral (Y likewise)
speed = |d/dt (X, Y, Z)|                   from a fitted trajectory
```

That recovers full 3D velocity from one camera, and means the ball can fly any
lane — depth is measured, not assumed.

Speed comes from **fitting the whole flight and differentiating the fit**,
never from frame-to-frame differences. Per-frame depth error at 10 m is around
30 cm, comparable to how far the ball travels between frames at 30 fps, so a
single pair of frames tells you almost nothing. Fitting ~15 of them reduces
the error by √N and yields roughly ±0.7 m/s.

## Setup

Requires Python 3.11+ and macOS (other platforms untested).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python fetch_models.py     # pretrained volleyball ball detector (~43 MB)
```

## Web UI

```bash
.venv/bin/python webapp.py
# http://127.0.0.1:5000
```

Upload a clip, watch the progress bar, get a table of peak speeds with a
quality flag on each, plus CSV export and an annotated video. Analysis runs in
a background thread, so the page stays responsive; jobs live in memory and are
lost when the server stops.

Analysis runs at roughly **1.4 fps**, so a 60-second 30 fps clip takes about
20 minutes. Use the "limit frames" box for a quick look.

## Command line

```bash
# one-time camera calibration: a ball of known apparent radius at a known distance
.venv/bin/python track.py --source clip.mp4 --calibrate 27.3 5.0

# measure speed
.venv/bin/python track.py --source clip.mp4 --speed \
    --ball-model models/volleyball_ball.pt --ball-conf 0.25 --ball-imgsz 1024

# just draw the ball, no speed
.venv/bin/python track.py --source clip.mp4 --save out.mp4 --no-display
```

### Calibration

`--calibrate RADIUS_PX DISTANCE_M` fits the camera's focal length and stores it
in `config.json`. It is per-device and never needs repeating. Without it the
app assumes a 70° field of view and says so — speeds are then indicative, and
scale linearly with the true focal length.

To get `RADIUS_PX`: film the ball held at a measured distance, run the tracker
on that clip, and read the reported radius.

**Calibration and measurement must use the same radius estimator.** The
bounding box is not the ball's true silhouette, but any constant bias divides
out exactly as long as both ends measure the same way — which is why
calibration goes through the same code path.

## Why the bounding box, not the segmentation mask

The model produces segmentation masks, and using them looks like the obvious
upgrade. Measured on real end-on footage, it is not:

| radius estimator | frame-to-frame noise | speed error at 12 m |
|---|---|---|
| **bounding-box max side** | **3.2%** | **±0.68 m/s (2.7%)** |
| mask area | 11.7% | ±2.51 m/s (10.0%) |
| mask minor axis | 20.3% | ±4.35 m/s (17.4%) |

Since depth error equals radius error, that is the difference between a usable
result and an unusable one. The masks go unused.

## Is it accurate?

```bash
.venv/bin/python tools/synthetic_validation.py
```

This composites a **real** volleyball, cut from real footage, onto a real gym
background along a trajectory specified to the millimetre, then runs the whole
production pipeline over it:

```
TRUE peak:     71.8 km/h
MEASURED peak: 72.3 km/h   (+0.7%, 26-point fit, ball found in 30/30 frames)
```

That validates detection → radius → depth → 3D fit → peak against known truth.
It does **not** reproduce motion blur or a real lens, so the gravity drop test
(TDD §8) is still required before trusting numbers from real footage.

## Detection

Stock COCO weights are not a volleyball detector — on sample footage the real
ball scored 0.13 while a player's shoe scored 0.15. `fetch_models.py` pulls a
YOLOv8n-seg model trained on volleyball (200 epochs, imgsz 1024, single `ball`
class) from
[masouduut94/volleyball_analytics](https://github.com/masouduut94/volleyball_analytics).

**Shape gating** (`--ball-max-size`, `--ball-max-aspect`) rejects candidates on
box geometry, since wall panels and lit windows are large and oblong while a
volleyball is small and round. On club-gym footage this cut candidates from 8.8
to 2.0 per frame.

**Static suppression** picks which candidate is the ball. Court lines, sponsor
banners and scoreboards never move, so anything reappearing in the same spot is
dropped and the most confident survivor wins. This replaced Roboflow's centroid
filter, which buffered *every* candidate and so got anchored onto exactly those
static objects — it reported a "ball" that moved a median of 0.9 px per frame.

| on 150 frames of club-gym footage | centroid | static |
|---|---|---|
| correct on hand-labelled ball frames | 2/5 | **4/5** |
| reports a ball when none is visible | 3/3 | **1/3** |
| median frame-to-frame motion | 0.9 px | **6.8 px** |

## Reading the results

Each flight segment gets a peak speed and a quality flag:

- **reliable** — 8+ points, complete arc. Trust it.
- **truncated** — the ball left frame mid-flight. The fit covers only part of
  the arc.
- **short** — too few points. A quadratic through 5 points fits anything; the
  peak it reports is partly noise.

A segment splits whenever the ball's direction changes sharply, so a
spike → block → floor sequence logs as three separate readings rather than one
merged number.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `clip.mp4` | Camera index or path to a video file |
| `--speed` | off | Measure speed rather than only drawing |
| `--calibrate R D` | — | Fit focal length from radius `R` px at distance `D` m |
| `--config` | `config.json` | Where intrinsics are stored |
| `--ball-model` | `yolo11n.pt` | Weights for ball detection |
| `--ball-conf` | `0.10` | Ball confidence floor, kept low on purpose |
| `--ball-imgsz` | `640` | Inference size per tile (`1024` for the volleyball model) |
| `--ball-max-size` | `60` | Reject candidates whose longer box side exceeds this (px) |
| `--ball-max-aspect` | `1.6` | Reject candidates less square than this ratio |
| `--ball-select` | `static` | `static` or `centroid` |
| `--static-window` | `25` | Frames of history used to judge stationarity |
| `--static-radius` | `18` | Reappearing within this many px counts as stationary |
| `--static-min-hits` | `6` | Past sightings inside the radius that mark it static |
| `--no-slice` | off | Detect on the full frame instead of tiles |
| `--max-frames N` | — | Stop after N frames |
| `--device` | auto | Torch device (`mps`, `cuda`, `cpu`) |
| `--save OUT.mp4` | off | Write annotated video here |

Run the tests with `.venv/bin/python -m pytest tests`.

## Known limitations

- **Detection recall is the ceiling, and on `input_videos/` it currently
  blocks measurement entirely.** The model finds *yellow* balls confidently
  (0.84) but never finds the white ball in play against grey concrete, even at
  confidence 0.05 with 3×3 tiling. Over a 250-frame window every detection was
  a static false positive at a fixed pixel position — net poles, not a ball.
  **Fine-tuning is required**, not optional: label end-on footage with white
  balls at far range, plus hard negatives for net poles and wall marks.
- **Static suppression assumes a roughly fixed camera.** Under a hard pan the
  court lines move, stop looking static, and survive the filter. Use a tripod.
- **A held ball is suppressed** — it is stationary. Harmless for speed, odd to
  watch during a timeout.
- **30 fps is tight.** A half-second flight gives ~15 samples. Filming at 60 or
  120 fps is the cheapest available accuracy win.
- **Printed balls are permanent false positives.** Sponsor banners and
  scoreboard graphics contain volleyball images, and a volleyball-specific
  model detects them *more* confidently than a generic one. They need to be
  labelled as hard negatives during fine-tuning.
- **No live mode.** At 1.4 fps this is an upload-and-wait tool.

## Recording good footage

- Camera **behind the server**, looking down the flight path.
- **Tripod.** Static suppression and every future camera-motion trick depend
  on it.
- Raised enough that the server's body does not hide the ball at contact.
- Frame tightly on the flight corridor so scoreboards, banners and windows sit
  outside the frame — removing false positives at capture time beats every
  algorithm here.
- **60 fps or higher** if the phone allows.
- Film one calibration clip per device: the ball held at a measured distance.

## Project files

| File | Purpose |
|---|---|
| `webapp.py` | Web UI — upload clips, poll progress, download results |
| `analysis.py` | End-to-end pipeline shared by the CLI and the web UI |
| `track.py` | CLI — tracking, speed measurement, calibration |
| `speed.py` | 3D trajectory fit, flight segmentation, peak speed |
| `calibration.py` | Camera intrinsics and depth from apparent size |
| `fetch_models.py` | Download the pretrained volleyball ball detector |
| `tracking/detectors.py` | YOLO backend, sliced ball detector, shape gate |
| `tracking/trackers.py` | Ball selection (static suppression), player ByteTrack |
| `tracking/annotators.py` | Ball trail and player ellipse overlays |
| `config.json` | Camera intrinsics, ball diameter |
| `PRD.md` / `TDD.md` | Product requirements and technical design |

## Roadmap

1. ~~**Detection & tracking**~~ — done: pretrained detector, shape gating,
   static suppression.
2. ~~**Speed on recorded clips**~~ — done: depth from ball size, 3D fit,
   per-flight peak, web UI.
3. **Validation**: gravity drop test and a known-distance roll test against
   the ±10% target; field check against a radar gun.
4. **Fine-tuning**: label end-on footage weighted toward far range, with hard
   negatives, to lift detection recall.
5. **Live mode**: real-time overlay, which needs a large speed-up first.
