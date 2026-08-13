"""Detection backends.

The ball is small and fast, so it is found with sliced inference: the frame
is cut into overlapping tiles, each tile is run through the detector at close
to native resolution, and the tile results are merged with non-max
suppression. A ball that occupies ~15px in a 1080p frame downscaled to 640
is only ~9px by the time it reaches the network; the same ball in a
1060x640 tile keeps its full size.

Players are large enough to survive downscaling, so they are detected on the
full frame — slicing them would cost 4x the inference for nothing.
"""
from __future__ import annotations

import numpy as np
import supervision as sv
from ultralytics import YOLO

# COCO class ids, used when running a stock (non-fine-tuned) model.
PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32


def resolve_class_ids(model: YOLO, coco_class_id: int) -> list[int] | None:
    """Which class ids to ask the model for.

    A stock COCO model needs filtering to the class we care about. A model
    fine-tuned on volleyball footage has its own small label set where the
    COCO ids are meaningless, so we take everything it reports instead.
    """
    names = model.names or {}
    if coco_class_id in names:
        return [coco_class_id]
    return None


class YoloBackend:
    """Ultralytics YOLO wrapped to return `sv.Detections`.

    Callable so it can be handed straight to `sv.InferenceSlicer`, which
    invokes it once per tile.
    """

    def __init__(self, model_path: str, coco_class_id: int,
                 confidence: float, imgsz: int = 640, device: str | None = None):
        self.model = YOLO(model_path)
        self.class_ids = resolve_class_ids(self.model, coco_class_id)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

    @property
    def is_finetuned(self) -> bool:
        """True when the weights aren't a stock COCO model."""
        return self.class_ids is None

    def __call__(self, image: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            image,
            conf=self.confidence,
            imgsz=self.imgsz,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)


class BallDetector:
    """Sliced ball detection.

    Slice size follows the Roboflow article: half the frame plus the overlap
    in each dimension, giving a 2x2 grid of overlapping tiles. The low IoU
    threshold is deliberate — the same ball seen in two overlapping tiles
    should collapse to one detection even when the two boxes agree only
    loosely.
    """

    def __init__(self, backend: YoloBackend, frame_wh: tuple[int, int],
                 overlap_wh: tuple[int, int] = (100, 100),
                 iou_threshold: float = 0.1, slice_inference: bool = True):
        self.backend = backend
        self.slice_inference = slice_inference
        width, height = frame_wh
        self.slicer = sv.InferenceSlicer(
            callback=backend,
            slice_wh=(width // 2 + overlap_wh[0], height // 2 + overlap_wh[1]),
            overlap_wh=overlap_wh,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            iou_threshold=iou_threshold,
        )

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        if not self.slice_inference:
            return self.backend(frame)
        return self.slicer(frame)


class PlayerDetector:
    """Full-frame player detection."""

    def __init__(self, backend: YoloBackend):
        self.backend = backend

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        return self.backend(frame)
