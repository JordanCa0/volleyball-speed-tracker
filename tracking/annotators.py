"""Overlay drawing.

The ball gets a motion trail rather than a box: at volleyball speeds a box
around a 15px ball is nearly invisible, while a fading trail of circles
shows both where it is and where it came from.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    """Ball motion trail (Roboflow ball-sports approach).

    Draws one circle per buffered frame, oldest smallest, coloured along
    matplotlib's "jet" ramp so the newest position reads hottest.
    """

    def __init__(self, radius: int = 12, buffer_size: int = 5, thickness: int = 2):
        self.color_palette = sv.ColorPalette.from_matplotlib("jet", buffer_size)
        self.buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """Grow the circle from 1px at the tail to `radius` at the head."""
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        xy = detections.get_anchors_coordinates(sv.Position.CENTER).astype(int)
        self.buffer.append(xy)

        for i, buffered_xy in enumerate(self.buffer):
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            color = self.color_palette.by_idx(i)
            for center in buffered_xy:
                cv2.circle(
                    img=frame,
                    center=(int(center[0]), int(center[1])),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )
        return frame

    def reset(self) -> None:
        self.buffer.clear()


class PlayerAnnotator:
    """Player ellipses at the feet, labelled with tracker id.

    An ellipse on the floor rather than a full box: it stays legible when
    players overlap, which they do constantly at the net.
    """

    def __init__(self, color: sv.Color | None = None):
        palette = sv.ColorPalette.DEFAULT
        self.ellipse = sv.EllipseAnnotator(color=color or palette, thickness=2)
        self.label = sv.LabelAnnotator(
            color=color or palette,
            text_color=sv.Color.BLACK,
            text_scale=0.4,
            text_padding=3,
            text_position=sv.Position.BOTTOM_CENTER,
        )

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        frame = self.ellipse.annotate(scene=frame, detections=detections)
        if detections.tracker_id is None:
            return frame
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        return self.label.annotate(scene=frame, detections=detections, labels=labels)
