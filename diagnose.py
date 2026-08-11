"""
Detection diagnostics (TDD §8 — measure before tuning).

Profiles a clip: how often is the ball found, and when it isn't, which
filter threw it away? Use this to decide what to fix instead of guessing.

Usage:
    python diagnose.py --source clip.mp4
    python diagnose.py --source clip.mp4 --sweep       # try looser settings
    python diagnose.py --source clip.mp4 --show 120    # eyeball the mask
"""
import argparse
import json

import cv2
import numpy as np

from color_ball import color_mask, detect_with_diagnostics


def profile(path, lower, upper, min_r, max_r, min_circ, use_motion=True,
            max_frames=None, stride=1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    sub = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False) \
        if use_motion else None

    stats = {"frames": 0, "found": 0, "no_candidates": 0, "too_small": 0,
             "too_large": 0, "not_circular": 0}
    radii, near_r, near_circ = [], [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and stats["frames"] >= max_frames):
            break
        idx += 1
        motion = None
        if sub is not None:
            motion = cv2.dilate(sub.apply(frame), None, iterations=4)
        if idx % stride:
            continue

        best, diag = detect_with_diagnostics(frame, lower, upper, min_r, max_r,
                                             motion, min_circ)
        stats["frames"] += 1
        if best:
            stats["found"] += 1
            radii.append(best[2])
        elif diag["candidates"] == 0:
            stats["no_candidates"] += 1
        else:
            # attribute the miss to whichever filter killed the best candidate
            if diag["too_small"]:
                stats["too_small"] += 1
                if diag["best_rejected_r"]:
                    near_r.append(diag["best_rejected_r"])
            elif diag["not_circular"]:
                stats["not_circular"] += 1
                if diag["best_rejected_circ"]:
                    near_circ.append(diag["best_rejected_circ"])
            elif diag["too_large"]:
                stats["too_large"] += 1
    cap.release()
    stats["radii"] = radii
    stats["near_r"] = near_r
    stats["near_circ"] = near_circ
    return stats


def report(stats, label=""):
    n = stats["frames"] or 1
    rate = 100 * stats["found"] / n
    print(f"{label}detected in {stats['found']}/{stats['frames']} frames ({rate:.1f}%)")
    misses = n - stats["found"]
    if misses:
        print(f"  misses by cause: "
              f"nothing ball-colored+moving {stats['no_candidates']} "
              f"({100*stats['no_candidates']/n:.0f}%), "
              f"too small {stats['too_small']} ({100*stats['too_small']/n:.0f}%), "
              f"not circular {stats['not_circular']} ({100*stats['not_circular']/n:.0f}%), "
              f"too large {stats['too_large']} ({100*stats['too_large']/n:.0f}%)")
    if stats["radii"]:
        r = np.array(stats["radii"])
        print(f"  detected radius px: min {r.min():.1f} median {np.median(r):.1f} max {r.max():.1f}")
    if stats["near_r"]:
        print(f"  largest blob when 'too small': median {np.median(stats['near_r']):.1f} px")
    if stats["near_circ"]:
        print(f"  best circularity when 'not circular': median {np.median(stats['near_circ']):.2f}")
    return rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--frames", type=int, default=600, help="frames to profile")
    parser.add_argument("--stride", type=int, default=1, help="profile every Nth frame")
    parser.add_argument("--sweep", action="store_true",
                        help="compare looser settings to find what's costing detections")
    parser.add_argument("--show", type=int, metavar="FRAME",
                        help="display the color/motion mask for one frame")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    lower, upper = config["hsv_lower"], config["hsv_upper"]
    r = config.get("ball_radius_px", config.get("min_radius_px", 8) * 2)
    min_r, max_r = max(r * 0.4, 3), r * 2.5

    if args.show is not None:
        cap = cv2.VideoCapture(args.source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.show)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"Could not read frame {args.show}")
        mask = color_mask(frame, lower, upper)
        stacked = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        h = 540
        scale = h / stacked.shape[0]
        cv2.imshow("frame | color mask (q to close)",
                   cv2.resize(stacked, None, fx=scale, fy=scale))
        while cv2.waitKey(30) & 0xFF != ord("q"):
            pass
        cv2.destroyAllWindows()
        return

    print(f"config: hsv {lower}..{upper}, radius {min_r:.1f}-{max_r:.1f}px\n")
    base = profile(args.source, lower, upper, min_r, max_r, 0.35,
                   max_frames=args.frames, stride=args.stride)
    report(base, "baseline: ")

    if args.sweep:
        print("\nsweep (what would each relaxation buy?)")
        variants = [
            ("no motion filter      ", dict(use_motion=False)),
            ("circularity 0.35->0.15", dict(min_circ=0.15)),
            ("min radius x0.5       ", dict(min_r=max(min_r * 0.5, 2))),
            ("wider hue (+/-10)     ", dict(lower=[max(lower[0] - 10, 0), max(lower[1] - 40, 20),
                                                   max(lower[2] - 40, 20)],
                                            upper=[min(upper[0] + 10, 179), 255, 255])),
            ("all of the above      ", dict(use_motion=False, min_circ=0.15,
                                            min_r=max(min_r * 0.5, 2),
                                            lower=[max(lower[0] - 10, 0), max(lower[1] - 40, 20),
                                                   max(lower[2] - 40, 20)],
                                            upper=[min(upper[0] + 10, 179), 255, 255])),
        ]
        for name, kw in variants:
            p = dict(lower=lower, upper=upper, min_r=min_r, max_r=max_r,
                     min_circ=0.35, use_motion=True, max_frames=args.frames,
                     stride=args.stride)
            p.update(kw)
            s = profile(args.source, p["lower"], p["upper"], p["min_r"], p["max_r"],
                        p["min_circ"], p["use_motion"], p["max_frames"], p["stride"])
            rate = 100 * s["found"] / (s["frames"] or 1)
            delta = rate - 100 * base["found"] / (base["frames"] or 1)
            print(f"  {name} {rate:5.1f}%  ({delta:+5.1f} pts)")


if __name__ == "__main__":
    main()
