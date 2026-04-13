"""Person candidate generation for Rule 1 pipelines."""

from __future__ import annotations

import io
from dataclasses import dataclass

from benchmark.constructionsite10k.types import ConstructionSiteSample
from common.schemas.bbox import NormalizedBBox


@dataclass(frozen=True, slots=True)
class PersonCandidate:
    """One detected person candidate used by Rule 1 inference."""

    candidate_id: str
    bbox: NormalizedBBox
    score: float


def pixel_xyxy_to_normalized_bbox(
    *,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    image_width: int,
    image_height: int,
) -> NormalizedBBox:
    """Convert a pixel-space detection box into normalized xyxy coordinates."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")

    return NormalizedBBox(
        x_min=max(0.0, min(1.0, x_min / image_width)),
        y_min=max(0.0, min(1.0, y_min / image_height)),
        x_max=max(0.0, min(1.0, x_max / image_width)),
        y_max=max(0.0, min(1.0, y_max / image_height)),
    )


class OpenCVHogPersonCandidateGenerator:
    """Generate person candidates with OpenCV's built-in HOG people detector."""

    def __init__(
        self,
        *,
        hit_threshold: float = 0.0,
        win_stride: tuple[int, int] = (8, 8),
        padding: tuple[int, int] = (8, 8),
        scale: float = 1.05,
        min_width: int = 24,
        min_height: int = 48,
    ) -> None:
        self._hit_threshold = hit_threshold
        self._win_stride = win_stride
        self._padding = padding
        self._scale = scale
        self._min_width = min_width
        self._min_height = min_height

    def generate(self, sample: ConstructionSiteSample) -> tuple[PersonCandidate, ...]:
        """Return normalized person candidates for one benchmark image."""
        rgb_image = _load_rgb_array(sample)
        image_height, image_width = rgb_image.shape[:2]

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - runtime dependency only
            raise ImportError("Rule 1 person detection requires opencv-python-headless.") from exc

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(
            rgb_image,
            hitThreshold=self._hit_threshold,
            winStride=self._win_stride,
            padding=self._padding,
            scale=self._scale,
        )

        candidates: list[PersonCandidate] = []
        for index, ((x, y, width, height), weight) in enumerate(
            zip(boxes, weights, strict=True),
            start=1,
        ):
            if width < self._min_width or height < self._min_height:
                continue
            bbox = pixel_xyxy_to_normalized_bbox(
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + width),
                y_max=float(y + height),
                image_width=image_width,
                image_height=image_height,
            )
            candidates.append(
                PersonCandidate(
                    candidate_id=f"person-{index}",
                    bbox=bbox,
                    score=float(weight),
                )
            )
        return tuple(candidates)


def _load_rgb_array(sample: ConstructionSiteSample):
    if sample.image is None or sample.image.bytes is None:
        raise ValueError(f"Sample {sample.image_id} does not contain embedded image bytes.")
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime dependency only
        raise ImportError("Rule 1 person detection requires numpy and Pillow.") from exc

    pil_image = Image.open(io.BytesIO(sample.image.bytes)).convert("RGB")
    return np.asarray(pil_image)
