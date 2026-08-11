"""
Speed measurement on recorded clips (M2).

Runs the full chain on a video: color ball detection -> gap-tolerant
tracking -> pixel->meter scaling -> live speed + per-hit peak speed.

Prerequisites:
- Ball color calibrated (run detect.py, press 'b', box the ball) so
  config.json has hsv bounds.
- A scale: either pass --calibrate-distance M to click two points that
  many meters apart on the first frame (court lines: attack->center = 3.0),
  or a previously saved scale in config.json is reused.

Usage:
    python track.py --source clip.mp4 --calibrate-distance 3.0
    python track.py --source clip.mp4 --save out.mp4 --csv hits.csv
"""
import argparse
import csv
import json
import time

import cv2

import calibration
from color_ball import find_candidates
from speed import SpeedEngine
from tracker import Detection, Tracker

ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)


def put_text(frame, text, org, scale=0.9):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="video file path")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--calibrate-distance", type=float, metavar="M",
                        help="click two points this many meters apart on the first frame")
    parser.add_argument("--max-speed-kmh", type=float, default=130.0,
                        help="tracker gate: fastest plausible ball")
    parser.add_argument("--threshold-kmh", type=float, default=15.0,
                        help="speed above which a hit segment opens")
    parser.add_argument("--csv", metavar="OUT.csv", help="write the hit log to this path")
    parser.add_argument("--save", metavar="OUT.mp4", help="write annotated video to this path")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    if "hsv_lower" not in config:
        raise SystemExit("No ball color in config — run detect.py, press 'b' and box the ball first.")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, first = cap.read()
    if not ok:
        raise SystemExit("Could not read the first frame.")

    if args.calibrate_distance:
        cal = calibration.interactive_two_point(first, args.calibrate_distance)
        calibration.save_to_config(cal, config)
        with open(args.config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Scale saved: {cal.meters_per_px:.5f} m/px ({cal.method})")
    else:
        cal = calibration.load_from_config(config)
        if cal is None and "ball_radius_px" in config:
            cal = calibration.from_ball_radius(config["ball_radius_px"],
                                               config.get("ball_diameter_m", 0.21))
            print(f"No saved scale; falling back to ball-diameter scaling "
                  f"({cal.meters_per_px:.5f} m/px) — prefer --calibrate-distance.")
        if cal is None:
            raise SystemExit("No scale available — pass --calibrate-distance M.")

    r = config.get("ball_radius_px", config.get("min_radius_px", 8) * 2)
    min_r, max_r = max(r * 0.4, 3), r * 2.5
    lower, upper = config["hsv_lower"], config["hsv_upper"]

    tracker = Tracker(max_speed_px_per_s=args.max_speed_kmh / 3.6 / cal.meters_per_px)
    engine = SpeedEngine(cal.meters_per_px, open_threshold_kmh=args.threshold_kmh)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False)

    writer = None
    if args.save:
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # replay from the start, including frame 0
    frame_idx = 0
    live = None
    t0 = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_idx / fps   # container timing: exact for files

            motion = cv2.dilate(subtractor.apply(frame), None, iterations=4)
            cands = find_candidates(frame, lower, upper, min_r, max_r, motion,
                                    expected_r=r)
            result = tracker.update_multi([
                Detection(frame_idx, timestamp, (c["x"], c["y"]), c["r"])
                for c in cands
            ])
            reading = engine.update(result.accepted, result.track_id, result.track_ended)
            if reading is not None:
                live = reading

            if writer or not args.no_display:
                if len(tracker.trail) > 1:
                    for a, b in zip(list(tracker.trail), list(tracker.trail)[1:]):
                        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                                 ORANGE, 2)
                if result.accepted:
                    x, y = result.accepted.center_px
                    cv2.circle(frame, (int(x), int(y)),
                               int(result.accepted.radius_px), ORANGE, 3)
                if live is not None:
                    put_text(frame, f"{live:5.1f} km/h ({live * 0.621:4.1f} mph)", (20, 45))
                if engine.last_hit:
                    put_text(frame, f"last hit: {engine.last_hit.peak_speed_kmh:.1f} km/h",
                             (20, 85), scale=0.8)
                if writer:
                    writer.write(frame)
                if not args.no_display:
                    cv2.imshow("track (q=quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            frame_idx += 1
    finally:
        engine.finish()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - t0
    print(f"\n{frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-9):.1f} fps)")
    if not engine.hits:
        print("No hits detected above the speed threshold.")
    else:
        print(f"{'hit':>4} {'start':>7} {'end':>7} {'peak km/h':>10} {'peak mph':>9}")
        for i, hit in enumerate(engine.hits, 1):
            print(f"{i:>4} {hit.start_time:>6.2f}s {hit.end_time:>6.2f}s "
                  f"{hit.peak_speed_kmh:>10.1f} {hit.peak_speed_kmh * 0.621:>9.1f}")

    if args.csv and engine.hits:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hit", "start_s", "end_s", "peak_kmh", "peak_mph"])
            for i, hit in enumerate(engine.hits, 1):
                w.writerow([i, f"{hit.start_time:.3f}", f"{hit.end_time:.3f}",
                            f"{hit.peak_speed_kmh:.1f}", f"{hit.peak_speed_kmh * 0.621:.1f}"])
        print(f"Hit log written to {args.csv}")


if __name__ == "__main__":
    main()
