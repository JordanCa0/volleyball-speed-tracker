# Volleyball Speed Tracker

Measure how fast a volleyball is hit (serves, spikes) using nothing but a
camera — no radar gun, no special hardware. See [PRD.md](PRD.md) for the
product vision and [TDD.md](TDD.md) for the technical design.

**Current status:** rebuilding detection and tracking on the approach from
Roboflow's [Tracking Ball Sports with Computer Vision][rf]. `track.py` finds
the ball and the players and draws them; **speed measurement is not currently
wired up** — `speed.py` holds the engine and its tests, but nothing feeds it
yet. That's the next milestone.

[rf]: https://blog.roboflow.com/tracking-ball-sports-computer-vision/

## How it works

```
frame ─┬─ 2x2 sliced inference ─ centroid filter ── ball trail overlay
       └─ full-frame inference ─ ByteTrack ─────── player ellipses + ids
```

The ball is detected with **sliced inference**: the frame is cut into
overlapping tiles and each is run through the detector at close to native
resolution. A ~15px ball in a 1080p frame downscaled to 640 is only ~9px by
the time it reaches the network; in a 1060x640 tile it keeps its full size.
On sample footage this raises the frames with a ball candidate from 35/60 to
51/60 and mean best confidence from 0.24 to 0.42.

Players are detected on the full frame — they survive downscaling, so slicing
them would cost 4x the inference for nothing.

## Setup

Requires Python 3.11+ and macOS (other platforms untested).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python fetch_models.py     # pretrained volleyball ball detector (~43 MB)
```

Stock YOLO weights download automatically on first run.

## The pretrained ball model

Stock COCO weights are not a volleyball detector — on sample broadcast
footage the real ball scored 0.13 while a player's shoe scored 0.15.
`fetch_models.py` pulls a YOLOv8n-seg model trained specifically on
volleyball (200 epochs, imgsz 1024, single `ball` class) from
[masouduut94/volleyball_analytics](https://github.com/masouduut94/volleyball_analytics).

```bash
.venv/bin/python track.py --source clip.mp4 \
    --ball-model models/volleyball_ball.pt --ball-conf 0.25 --ball-imgsz 1024
```

Measured against stock COCO over 120 frames of broadcast footage:

| | stock COCO | volleyball model |
|---|---|---|
| frames with a ball candidate | 92% | **100%** |
| mean best confidence | 0.41 | **0.85** |

It is a large gain in *recall* and a smaller one in precision. The model
also fires confidently on printed volleyballs (sponsor banners, scoreboard
graphics), and in a club gym with lime-green walls it fires on wall panels,
lit windows and ceiling fixtures — roughly 9 candidates per frame. The shape
gate below is what makes that usable.

### Shape gating

A volleyball is round and small; the things a ball detector confuses it with
usually aren't. `--ball-max-size` and `--ball-max-aspect` reject candidates
on box geometry before they reach the tracker:

| footage | candidates/frame, no gate | with 60px / 1.6 gate |
|---|---|---|
| broadcast | 6.0 | **2.4** |
| club gym | 8.8 | **2.0** |

On one club-gym frame this cut 17 candidates to 2, keeping the real ball as
the highest-confidence detection. The cost is recall: 7 of 80 club-gym
frames lost their only candidate. Raise `--ball-max-size` if your ball is
larger in frame (closer camera, higher resolution).

## Usage

```bash
# watch a clip
.venv/bin/python track.py --source clip.mp4

# save an annotated copy, no window
.venv/bin/python track.py --source clip.mp4 --save out.mp4 --no-display

# quick experiment on the first 300 frames
.venv/bin/python track.py --source clip.mp4 --max-frames 300

# compare sliced vs full-frame ball detection
.venv/bin/python track.py --source clip.mp4 --no-slice

# once you have fine-tuned weights
.venv/bin/python track.py --source clip.mp4 --ball-model models/ball_best.pt
```

Press `q` to quit the live window.

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `clip.mp4` | Camera index or path to a video file |
| `--ball-model` | `yolo11n.pt` | Weights for ball detection |
| `--player-model` | `yolo11n.pt` | Weights for player detection |
| `--ball-conf` | `0.10` | Ball confidence floor, kept low on purpose |
| `--player-conf` | `0.30` | Player confidence floor |
| `--ball-imgsz` | `640` | Inference size per tile (use `1024` for the volleyball model) |
| `--ball-max-size` | `60` | Reject candidates whose longer box side exceeds this (px); `0` disables |
| `--ball-max-aspect` | `1.6` | Reject candidates less square than this ratio; `0` disables |
| `--player-imgsz` | `960` | Inference size for the full frame |
| `--buffer-size` | `10` | Frames of candidate history for the ball filter |
| `--trail-length` | `5` | Frames of ball trail to draw |
| `--no-slice` | off | Detect the ball on the full frame instead of tiles |
| `--no-ball` / `--no-players` | off | Skip one half of the pipeline |
| `--max-frames N` | — | Stop after N frames |
| `--device` | auto | Torch device (`mps`, `cuda`, `cpu`) |
| `--save OUT.mp4` | off | Write annotated video here |
| `--no-display` | off | Don't open the live window |

Expect roughly **4 fps at 1080p** — five forward passes per frame (four ball
tiles plus one player pass). Use `--max-frames` while iterating.

Run the test suite with `.venv/bin/python -m pytest tests`.

## Known limitations

- **Stock COCO weights are unusable for the ball** — use the pretrained
  volleyball model above.
- **The pretrained model is out of distribution on club-gym footage.**
  Lime-green wall panels and lit windows read as balls. Shape gating removes
  most of it; fine-tuning on your own footage is the real fix.
- **Printed balls are permanent false positives.** Sponsor banners and
  scoreboard graphics contain volleyball images, and a volleyball-specific
  model detects them *more* confidently than a generic one (0.79 on a
  scoreboard graphic). They never move, so no temporal filter removes them;
  they need to be labelled as hard negatives during fine-tuning.
- **Camera motion breaks static-background tricks.** Measured over 400
  frames: broadcast footage drifts 44px total (near-locked), but club
  handheld footage pans 1017px and zooms 5.7%. Background subtraction and
  fixed court polygons both require motion compensation on the latter — and
  so will any speed measurement, since a 54px/frame pan swamps the ball's own
  displacement.
- **No court-boundary filtering.** Benches, coaches, referees and the front
  row of the crowd are all tracked as players.
- **The ball filter lags.** It picks the candidate nearest the centroid of
  recent positions, which sits behind a fast-moving ball. Harmless when the
  ball is the only candidate, costly when it isn't. Tune with `--buffer-size`.

## Recording good footage

For best results (per [TDD.md §3](TDD.md)):

- Place the camera **side-on** to the flight path — perpendicular, not
  behind the server.
- Stand **6–10 m back** so the whole flight fits in frame.
- Prefer the phone's main 1× lens over 0.5× ultra-wide (less distortion,
  bigger ball in pixels) — back up instead of zooming out.
- Higher frame rates (60 fps+) reduce motion blur on fast hits.

## Project files

| File | Purpose |
|---|---|
| `track.py` | CLI entry point — detect and track ball + players |
| `fetch_models.py` | Download the pretrained volleyball ball detector |
| `tracking/detectors.py` | YOLO backend, sliced ball detector, player detector |
| `tracking/trackers.py` | Ball centroid filter, ByteTrack player tracker |
| `tracking/annotators.py` | Ball trail and player ellipse overlays |
| `speed.py` | Per-hit peak speed engine (not currently wired up) |
| `calibration.py` | Pixel→metre scale calibration (not currently wired up) |
| `config.json` | Camera index, ball diameter, saved scale |
| `PRD.md` | Product requirements — what we're building and why |
| `TDD.md` | Technical design — architecture, algorithms, milestones |

## Roadmap

1. **Detection & tracking** *(current)*: reliably find the ball and players.
   Next up: fine-tune a ball detector on labelled volleyball footage, and add
   court-boundary filtering so off-court people are ignored.
2. **Speed on recorded clips**: re-wire `speed.py` to the new pipeline;
   pixel→real-world conversion, per-hit peak speed, validated against gravity
   drop tests.
3. **Live mode**: real-time speed overlay from a webcam, session log with CSV
   export.
4. **Accuracy pass**: court-keypoint calibration, high-fps input, field
   validation against a radar gun.
