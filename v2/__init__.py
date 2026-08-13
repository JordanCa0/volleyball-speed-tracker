"""Volleyball tracker v2 — ball and player tracking.

Follows the approach in Roboflow's "Tracking Ball Sports with Computer
Vision": sliced inference to find a small fast ball, a centroid-buffer
filter to suppress false positives, and ByteTrack for the players.

Speed measurement is deliberately out of scope here.
"""
from .detectors import BallDetector, PlayerDetector, YoloBackend
from .trackers import BallTracker, PlayerTracker
from .annotators import BallAnnotator, PlayerAnnotator

__all__ = [
    "BallDetector",
    "PlayerDetector",
    "YoloBackend",
    "BallTracker",
    "PlayerTracker",
    "BallAnnotator",
    "PlayerAnnotator",
]
