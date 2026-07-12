"""
Player + ball detector (M1 groundwork).

Runs a pretrained YOLO model over a video file or live camera and draws
labeled boxes around every detected person and sports ball.

Usage:
    python detect.py --source path/to/clip.mp4
    python detect.py --source 0                # webcam
    python detect.py --source clip.mp4 --save out.mp4 --no-display

Press 'q' to quit the display window.
"""
import argparse
import time

import cv2
from ultralytics import YOLO

# COCO class ids the pretrained model was trained on
PERSON = 0
SPORTS_BALL = 32

BOX_STYLE = {
    PERSON: ("player", (80, 200, 80)),      # green (BGR)
    SPORTS_BALL: ("ball", (0, 140, 255)),   # orange
}


def draw_detections(frame, result):
    for box in result.boxes:
        cls = int(box.cls)
        label, color = BOX_STYLE[cls]
        conf = float(box.conf)
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        thickness = 3 if cls == SPORTS_BALL else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        caption = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index or video file path")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="ultralytics model weights (n=nano fastest, s/m/l more accurate)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="minimum detection confidence")
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

    model = YOLO(args.model)  # downloads weights on first run

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
                                   conf=args.conf, verbose=False)[0]
            draw_detections(frame, result)
            frames += 1

            if writer:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("detect (q=quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
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
