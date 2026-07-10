"""
Interactive HSV threshold tuner.

Point this at your camera (or a recorded clip) with the ball visible,
drag the trackbars until only the ball shows up white in the "mask" window,
then press 's' to save those bounds into config.json.
"""
import argparse
import json

import cv2
import numpy as np

WINDOW = "HSV calibration (s=save, q=quit)"


def nothing(_):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="camera index or video file path")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    with open(args.config) as f:
        config = json.load(f)

    cv2.namedWindow(WINDOW)
    lo, hi = config["hsv_lower"], config["hsv_upper"]
    cv2.createTrackbar("H min", WINDOW, lo[0], 179, nothing)
    cv2.createTrackbar("H max", WINDOW, hi[0], 179, nothing)
    cv2.createTrackbar("S min", WINDOW, lo[1], 255, nothing)
    cv2.createTrackbar("S max", WINDOW, hi[1], 255, nothing)
    cv2.createTrackbar("V min", WINDOW, lo[2], 255, nothing)
    cv2.createTrackbar("V max", WINDOW, hi[2], 255, nothing)

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h_min = cv2.getTrackbarPos("H min", WINDOW)
        h_max = cv2.getTrackbarPos("H max", WINDOW)
        s_min = cv2.getTrackbarPos("S min", WINDOW)
        s_max = cv2.getTrackbarPos("S max", WINDOW)
        v_min = cv2.getTrackbarPos("V min", WINDOW)
        v_max = cv2.getTrackbarPos("V max", WINDOW)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        stacked = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
        cv2.imshow(WINDOW, stacked)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            config["hsv_lower"] = [int(h_min), int(s_min), int(v_min)]
            config["hsv_upper"] = [int(h_max), int(s_max), int(v_max)]
            with open(args.config, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Saved to {args.config}: lower={config['hsv_lower']} upper={config['hsv_upper']}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
