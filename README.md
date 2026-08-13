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
```

The first run downloads the YOLO weights (~6 MB) automatically.

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
| `--ball-imgsz` | `640` | Inference size per tile |
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

- **Stock COCO weights are the bottleneck.** On sample footage the real ball
  scores ~0.13 confidence while a player's shoe scores 0.15, and the ball is
  missed entirely in frames where it is plainly visible. Fine-tuning a ball
  detector is the next piece of work.
- **Printed balls are permanent false positives.** Sponsor banners and
  scoreboard graphics in broadcast footage contain volleyball images. They
  never move, so no temporal filter removes them; they need to be labelled as
  hard negatives during fine-tuning.
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
