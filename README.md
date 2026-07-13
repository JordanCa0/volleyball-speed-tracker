# Volleyball Speed Tracker

Measure how fast a volleyball is hit (serves, spikes) using nothing but a
camera — no radar gun, no special hardware. See [PRD.md](PRD.md) for the
product vision and [TDD.md](TDD.md) for the technical design.

**Current status:** early prototype. The detector (`detect.py`) watches
footage and draws boxes around players and the ball. Speed measurement is
the next milestone.

## Setup

Requires Python 3.11+ and macOS (other platforms untested).

```bash
# from the project root
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run of `detect.py` automatically downloads the YOLO model
weights (~6 MB).

## Usage

### Detect players and the ball in a video file

```bash
.venv/bin/python detect.py --source path/to/clip.mp4
```

A window opens showing the footage with a green box around each detected
player and an orange box around the ball, each labeled with a confidence
percentage. Press `q` to quit.

### Live webcam

```bash
.venv/bin/python detect.py --source 1
```

The number is the camera index — see
[Finding your camera index](#finding-your-camera-index) below.

### Save an annotated copy

```bash
.venv/bin/python detect.py --source clip.mp4 --save annotated.mp4 --no-display
```

`--no-display` skips the live window (useful for batch processing);
`--save` writes the boxed-up video to a file.

### All options

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `0` | Camera index or path to a video file |
| `--model` | `yolo11n.pt` | YOLO weights; `n` (nano) is fastest, `s`/`m`/`l` are more accurate but slower |
| `--conf` | `0.4` | Minimum confidence for player detections |
| `--ball-conf` | `0.12` | Minimum confidence for YOLO ball detections (kept low — blurry balls score low) |
| `--imgsz` | `960` | Inference resolution; higher finds smaller balls but runs slower |
| `--use-color` | off | Enable color-based ball detection using bounds saved in config |
| `--no-motion-filter` | off | Don't require the color-detected ball to be moving |
| `--config` | `config.json` | Where calibration is loaded from / saved to |
| `--save OUT.mp4` | off | Write annotated video to this path |
| `--no-display` | off | Don't open the live window |

**Teaching it your ball:** while the video plays, press `b`, draw a box
around the ball, and press ENTER. The ball's color is sampled and saved to
the config; a color-based detector then finds the ball even when YOLO
misses it. By default a detected ball must also be *moving* — this rejects
static ball-colored objects (broadcast GUI overlays, floors, walls).

## Finding your camera index

On macOS, camera index 0 is often an iPhone (Continuity Camera) rather
than the built-in webcam — a locked iPhone shows up as a black feed.
Probe which index is which:

```bash
.venv/bin/python -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    ok, frame = (cap.read() if cap.isOpened() else (False, None))
    print(f'camera {i}:', f'{frame.shape[1]}x{frame.shape[0]}, brightness {frame.mean():.0f}' if ok else 'no frames')
    cap.release()
"
```

A working camera reports a resolution and non-zero brightness; the black
iPhone feed reports brightness 0.

**No frames at all?** Grant camera permission to your terminal app in
**System Settings → Privacy & Security → Camera**, then fully quit and
reopen the terminal.

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
| `detect.py` | YOLO player + ball detector (current prototype) |
| `hsv_calibrate.py` | Interactive color-threshold tuner (alternative ball-detection approach) |
| `config.json` | Camera index, color bounds, ball diameter |
| `PRD.md` | Product requirements — what we're building and why |
| `TDD.md` | Technical design — architecture, algorithms, milestones |

## Roadmap

Following the milestones in [TDD.md §7](TDD.md):

1. **M1 — Detection** *(in progress)*: reliably detect and track the ball
   in recorded clips.
2. **M2 — Speed on recorded clips**: pixel→real-world conversion,
   per-hit peak speed, validated against gravity drop tests.
3. **M3 — Live mode**: real-time speed overlay from a webcam, session log
   with CSV export.
4. **M4 — Accuracy pass**: court-line homography calibration, high-fps
   input, field validation against a radar gun.
