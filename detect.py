"""
Player + ball detector (M1 groundwork).

Runs a pretrained YOLO model over a video file or live camera and draws
labeled boxes around every detected person and sports ball.

Because a small, motion-blurred volleyball is hard for a generic YOLO
model, you can teach the program your ball's color: press 'b' during
playback, draw a box around the ball, press ENTER. From then on a
color-based detector runs alongside YOLO for the ball, and the sampled
color is saved to config.json (reuse it later with --use-color).

Usage:
    python detect.py --source path/to/clip.mp4
    python detect.py --source 1                 # webcam
    python detect.py --source clip.mp4 --use-color --save out.mp4 --no-display

Keys: 'b' = select the ball to calibrate color, 'q' = quit.
"""
import argparse
import json
import time

import cv2
from ultralytics import YOLO

from color_ball import detect_ball_by_color, sample_ball_color

# COCO class ids the pretrained model was trained on
PERSON = 0
SPORTS_BALL = 32

BOX_STYLE = {
    PERSON: ("player", (80, 200, 80)),      # green (BGR)
    SPORTS_BALL: ("ball", (0, 140, 255)),   # orange
}


def draw_box(frame, x1, y1, x2, y2, caption, color, thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, caption, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_yolo_detections(frame, result, person_conf, ball_conf, skip_ball):
    for box in result.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        threshold = ball_conf if cls == SPORTS_BALL else person_conf
        if conf < threshold or (cls == SPORTS_BALL and skip_ball):
            continue
        label, color = BOX_STYLE[cls]
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        draw_box(frame, x1, y1, x2, y2, f"{label} {conf:.0%}", color,
                 thickness=3 if cls == SPORTS_BALL else 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index or video file path")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="ultralytics model weights (n=nano fastest, s/m/l more accurate)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="minimum confidence for player detections")
    parser.add_argument("--ball-conf", type=float, default=0.12,
                        help="minimum confidence for YOLO ball detections (kept low on purpose)")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="inference resolution; higher finds smaller balls, slower")
    parser.add_argument("--use-color", action="store_true",
                        help="enable color ball detection from bounds saved in config")
    parser.add_argument("--no-motion-filter", action="store_true",
                        help="don't require the color-detected ball to be moving")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--save", metavar="OUT.mp4", help="write annotated video to this path")
    parser.add_argument("--no-display", action="store_true", help="skip the live window")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    is_camera = isinstance(source, int)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    if is_camera:
        # Cameras can take a moment to deliver the first frame; retry briefly.
        deadline = time.monotonic() + 5.0
        ok = False
        while time.monotonic() < deadline:
            ok, _ = cap.read()
            if ok:
                break
        if not ok:
            raise SystemExit(
                f"Camera {source} opened but delivered no frames.\n"
                "On macOS this usually means:\n"
                "  - Terminal lacks camera permission: System Settings -> "
                "Privacy & Security -> Camera -> enable your terminal app, then restart it\n"
                f"  - Index {source} is an inactive device (e.g. iPhone Continuity "
                f"Camera): try --source {source + 1}"
            )

    with open(args.config) as f:
        config = json.load(f)

    color_bounds = None   # (lower, upper, min_r, max_r) when color detection is active
    if args.use_color:
        r = config.get("ball_radius_px", config.get("min_radius_px", 8) * 2)
        color_bounds = (config["hsv_lower"], config["hsv_upper"], max(r * 0.4, 3), r * 2.5)
        print(f"Color ball detection on: hsv {config['hsv_lower']}..{config['hsv_upper']}")

    model = YOLO(args.model)  # downloads weights on first run

    # Learns the static background so moving pixels can be isolated; runs on
    # every frame (cheap) so it's already warmed up when 'b' is pressed.
    subtractor = None if args.no_motion_filter else \
        cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False)

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames = 0
    t0 = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(frame, classes=[PERSON, SPORTS_BALL],
                                   conf=min(args.conf, args.ball_conf),
                                   imgsz=args.imgsz, verbose=False)[0]

            motion_mask = None
            if subtractor is not None:
                motion_mask = subtractor.apply(frame)
                # Dilate so a fast ball's slightly-lagging motion blob still
                # covers its color blob.
                motion_mask = cv2.dilate(motion_mask, None, iterations=4)

            ball = None
            if color_bounds:
                lower, upper, min_r, max_r = color_bounds
                ball = detect_ball_by_color(frame, lower, upper, min_r, max_r,
                                            motion_mask)
                if ball:
                    x, y, r = ball
                    cv2.circle(frame, (int(x), int(y)), int(r), (0, 140, 255), 2)
                    draw_box(frame, int(x - r), int(y - r), int(x + r), int(y + r),
                             "ball (color)", (0, 140, 255), thickness=3)

            draw_yolo_detections(frame, result, args.conf, args.ball_conf,
                                 skip_ball=ball is not None)
            frames += 1

            if writer:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("detect (b=select ball, q=quit)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    roi = cv2.selectROI("draw a box around the ball, ENTER to confirm",
                                        frame, showCrosshair=True)
                    cv2.destroyWindow("draw a box around the ball, ENTER to confirm")
                    if roi[2] > 0 and roi[3] > 0:
                        lower, upper = sample_ball_color(frame, roi)
                        r_sel = max(roi[2], roi[3]) / 2
                        color_bounds = (lower, upper, max(r_sel * 0.4, 3), r_sel * 2.5)
                        config["hsv_lower"], config["hsv_upper"] = lower, upper
                        config["ball_radius_px"] = round(r_sel, 1)
                        with open(args.config, "w") as f:
                            json.dump(config, f, indent=2)
                        print(f"Ball color calibrated: hsv {lower}..{upper}, "
                              f"radius ~{r_sel:.0f}px (saved to {args.config})")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - t0
    print(f"{frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} fps)")
    if args.save:
        print(f"Annotated video written to {args.save}")


if __name__ == "__main__":
    main()
