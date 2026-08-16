"""Ball and player tracking.

Implements the pipeline from Roboflow's "Tracking Ball Sports with Computer
Vision":

    frame -> sliced inference (ball) -> centroid filter -> trail overlay
          -> full-frame inference (players) -> ByteTrack -> ellipse overlay

Speed measurement is not part of this; getting the detections and identities
right comes first. See speed.py for the (currently unwired) speed engine.

Usage:
    python track.py --source clip.mp4
    python track.py --source clip.mp4 --save out.mp4 --no-display
    python track.py --source clip.mp4 --max-frames 300 --no-slice
    python track.py --source clip.mp4 --ball-model models/ball_best.pt

Keys: 'q' = quit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import supervision as sv

from analysis import analyse_video
from calibration import focal_length_from_reference, load_from_config, save_to_config
from tracking.detectors import (
    PERSON_CLASS_ID,
    SPORTS_BALL_CLASS_ID,
    BallDetector,
    PlayerDetector,
    YoloBackend,
)
from tracking.trackers import BallTracker, PlayerTracker
from tracking.annotators import BallAnnotator, PlayerAnnotator

WINDOW = "volleyball tracker (q=quit)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="clip.mp4",
                        help="video file path or camera index")
    parser.add_argument("--ball-model", default="yolo11n.pt",
                        help="weights for ball detection; point at fine-tuned "
                             "weights once you have them")
    parser.add_argument("--player-model", default="yolo11n.pt",
                        help="weights for player detection")
    parser.add_argument("--ball-conf", type=float, default=0.10,
                        help="ball confidence floor, kept low on purpose — the "
                             "centroid filter is what rejects false positives")
    parser.add_argument("--player-conf", type=float, default=0.30)
    parser.add_argument("--ball-imgsz", type=int, default=640,
                        help="inference size per tile")
    parser.add_argument("--player-imgsz", type=int, default=960)
    parser.add_argument("--ball-max-size", type=float, default=60.0,
                        help="reject ball candidates whose longer box side "
                             "exceeds this many pixels (0 disables)")
    parser.add_argument("--ball-max-aspect", type=float, default=1.6,
                        help="reject ball candidates less square than this "
                             "long/short side ratio (0 disables)")
    parser.add_argument("--ball-select", choices=("static", "centroid"),
                        default="static",
                        help="how to pick the ball among candidates: 'static' "
                             "drops things that never move then takes the most "
                             "confident; 'centroid' is the original Roboflow "
                             "filter (kept for comparison)")
    parser.add_argument("--static-window", type=int, default=25,
                        help="frames of history used to judge whether a "
                             "candidate is stationary")
    parser.add_argument("--static-radius", type=float, default=18.0,
                        help="a candidate counts as stationary if it keeps "
                             "reappearing within this many pixels")
    parser.add_argument("--static-min-hits", type=int, default=6,
                        help="how many past sightings inside --static-radius "
                             "mark a candidate as stationary")
    parser.add_argument("--buffer-size", type=int, default=10,
                        help="frames of candidate history for the centroid "
                             "strategy; lower follows a fast ball better, "
                             "higher rejects more false positives")
    parser.add_argument("--trail-length", type=int, default=5,
                        help="frames of ball trail to draw")
    parser.add_argument("--no-slice", action="store_true",
                        help="detect the ball on the full frame instead of "
                             "tiles (much faster, finds fewer small balls)")
    parser.add_argument("--no-ball", action="store_true")
    parser.add_argument("--no-players", action="store_true")
    parser.add_argument("--max-frames", type=int,
                        help="stop after N frames (for quick experiments)")
    parser.add_argument("--device", help="torch device, e.g. mps / cuda / cpu")
    parser.add_argument("--save", metavar="OUT.mp4", help="write annotated video here")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--speed", action="store_true",
                        help="measure ball speed instead of just drawing: "
                             "recovers depth from the ball's apparent size and "
                             "reports a peak speed per flight segment")
    parser.add_argument("--calibrate", nargs=2, type=float,
                        metavar=("RADIUS_PX", "DISTANCE_M"),
                        help="fit the camera focal length from a ball of known "
                             "apparent radius at a known distance, save to "
                             "config.json, and exit")
    parser.add_argument("--config", default="config.json",
                        help="where camera intrinsics are stored")
    return parser


def open_source(source: str) -> tuple[cv2.VideoCapture, sv.VideoInfo]:
    handle = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(handle)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    return capture, sv.VideoInfo(width=width, height=height, fps=fps, total_frames=total)


def run_calibration(args) -> None:
    """Fit and persist the focal length from one reference measurement."""
    radius_px, distance_m = args.calibrate
    capture, video_info = open_source(args.source)
    capture.release()
    intrinsics = focal_length_from_reference(
        radius_px, distance_m, (video_info.width, video_info.height))

    path = Path(args.config)
    config = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(save_to_config(intrinsics, config), indent=2))

    print(f"focal length: {intrinsics.focal_px:.1f} px "
          f"(from r={radius_px}px at {distance_m}m)")
    print(f"saved to {path}")


def run_speed(args) -> None:
    """Measure speed rather than just drawing the ball."""
    path = Path(args.config)
    config = json.loads(path.read_text()) if path.exists() else {}
    intrinsics = load_from_config(config)
    if intrinsics is None:
        print("no calibration found — falling back to an assumed 70° field of "
              "view. Speeds are indicative only; run --calibrate for real "
              "numbers.")

    result = analyse_video(
        args.source, intrinsics=intrinsics, ball_model=args.ball_model,
        confidence=args.ball_conf, imgsz=args.ball_imgsz,
        max_side_px=args.ball_max_size or None,
        max_aspect=args.ball_max_aspect or None,
        slice_inference=not args.no_slice, max_frames=args.max_frames,
        device=args.device, annotate_path=args.save,
        progress=lambda n, total: print(f"  {n}/{total or '?'} frames", flush=True)
        if n % 100 == 0 else None,
    )

    print()
    print(result.summary())
    print(f"scale source: {result.intrinsics_source}")
    if not result.hits:
        print("\nno flight segments found")
        return
    print(f"\n{'#':>3}  {'start':>7}  {'peak km/h':>9}  {'mph':>6}  "
          f"{'pts':>4}  {'depth m':>7}  note")
    for i, hit in enumerate(result.hits, 1):
        note = "" if hit.reliable else ("truncated" if hit.truncated else "short")
        print(f"{i:>3}  {hit.start_time:>6.2f}s  {hit.peak_speed_kmh:>9.1f}  "
              f"{hit.peak_speed_kmh*0.621371:>6.1f}  {hit.n_points:>4}  "
              f"{hit.mean_depth_m:>7.1f}  {note}")
    if result.fastest:
        print(f"\nfastest: {result.fastest.peak_speed_kmh:.1f} km/h "
              f"({result.fastest.peak_speed_kmh*0.621371:.1f} mph)")
    if args.save:
        print(f"wrote {args.save}")


def main() -> None:
    args = build_parser().parse_args()
    if args.calibrate:
        run_calibration(args)
        return
    if args.speed:
        run_speed(args)
        return

    capture, video_info = open_source(args.source)
    print(f"source: {args.source}  {video_info.width}x{video_info.height} "
          f"@ {video_info.fps:.2f}fps  frames={video_info.total_frames}")

    ball_detector = ball_tracker = ball_annotator = None
    if not args.no_ball:
        ball_backend = YoloBackend(args.ball_model, SPORTS_BALL_CLASS_ID,
                                   args.ball_conf, args.ball_imgsz, args.device)
        ball_detector = BallDetector(
            ball_backend,
            frame_wh=(video_info.width, video_info.height),
            slice_inference=not args.no_slice,
            max_side_px=args.ball_max_size or None,
            max_aspect=args.ball_max_aspect or None,
        )
        ball_tracker = BallTracker(
            buffer_size=args.buffer_size,
            strategy=args.ball_select,
            static_window=args.static_window,
            static_radius=args.static_radius,
            static_min_hits=args.static_min_hits,
        )
        ball_annotator = BallAnnotator(radius=12, buffer_size=args.trail_length)
        mode = "full-frame" if args.no_slice else "2x2 sliced"
        weights = "fine-tuned" if ball_backend.is_finetuned else "stock COCO"
        print(f"ball:    {args.ball_model} ({weights}), {mode}, "
              f"conf={args.ball_conf}, select={args.ball_select}")

    player_detector = player_tracker = player_annotator = None
    if not args.no_players:
        player_backend = YoloBackend(args.player_model, PERSON_CLASS_ID,
                                     args.player_conf, args.player_imgsz, args.device)
        player_detector = PlayerDetector(player_backend)
        player_tracker = PlayerTracker(frame_rate=video_info.fps)
        player_annotator = PlayerAnnotator()
        print(f"players: {args.player_model}, conf={args.player_conf}")

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 video_info.fps, (video_info.width, video_info.height))

    frames = 0
    frames_with_ball_candidate = 0
    started = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if ball_detector is not None:
                ball_detections = ball_detector(frame)
                ball_detections = ball_tracker.update(ball_detections)
                frames_with_ball_candidate += len(ball_detections)
                frame = ball_annotator.annotate(frame, ball_detections)

            if player_detector is not None:
                player_detections = player_detector(frame)
                player_detections = player_tracker.update(player_detections)
                frame = player_annotator.annotate(frame, player_detections)

            frames += 1
            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if frames % 50 == 0:
                rate = frames / (time.monotonic() - started)
                print(f"  {frames} frames  ({rate:.1f} fps)", flush=True)
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    print(f"\n{frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-9):.1f} fps)")
    if ball_detector is not None:
        # Not an accuracy figure: this counts frames where the filter emitted
        # something, and with stock COCO weights that something is regularly a
        # shoe or a printed ball on a sponsor banner. Only a fine-tuned model
        # makes this number mean "found the ball".
        print(f"ball candidate emitted in {frames_with_ball_candidate}/{frames} "
              f"frames ({100 * frames_with_ball_candidate / max(frames, 1):.0f}%)")
    if args.save:
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
