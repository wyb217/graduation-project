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


class TorchvisionPersonCandidateGenerator:
    """Generate person candidates with a torchvision COCO detector."""

    def __init__(
        self,
        *,
        person_detector: object | None = None,
        score_threshold: float = 0.3,
        min_width: int = 24,
        min_height: int = 48,
        person_label: int = 1,
    ) -> None:
        self._person_detector = person_detector
        self._score_threshold = score_threshold
        self._min_width = min_width
        self._min_height = min_height
        self._person_label = person_label
        self._torch = None
        self._device = None

    def generate(self, sample: ConstructionSiteSample) -> tuple[PersonCandidate, ...]:
        """Return normalized person candidates for one benchmark image."""
        image = _load_pil_image(sample)
        image_width, image_height = image.size
        detections = self._detect(image)

        boxes = _as_python_list(detections.get("boxes", []))
        labels = _as_python_list(detections.get("labels", []))
        scores = _as_python_list(detections.get("scores", []))

        candidates: list[PersonCandidate] = []
        for box, label, score in zip(boxes, labels, scores, strict=False):
            if int(label) != self._person_label or float(score) < self._score_threshold:
                continue
            x_min, y_min, x_max, y_max = [float(value) for value in box]
            width = x_max - x_min
            height = y_max - y_min
            if width < self._min_width or height < self._min_height:
                continue
            bbox = pixel_xyxy_to_normalized_bbox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                image_width=image_width,
                image_height=image_height,
            )
            candidates.append(
                PersonCandidate(
                    candidate_id=f"person-{len(candidates) + 1}",
                    bbox=bbox,
                    score=float(score),
                )
            )
        return tuple(candidates)

    def _detect(self, image) -> dict[str, object]:
        detector = self._get_detector()
        if self._torch is None:
            outputs = detector([image])
            return outputs[0]

        tensor_image = self._pil_to_tensor(image).to(self._device)
        with self._torch.no_grad():
            outputs = detector([tensor_image])
        return outputs[0]

    def _get_detector(self):
        if self._person_detector is not None:
            return self._person_detector

        try:
            import numpy as np
            import torch
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_V2_Weights,
                fasterrcnn_resnet50_fpn_v2,
            )
        except ImportError as exc:  # pragma: no cover - runtime dependency only
            raise ImportError(
                "Torchvision person fallback requires torch and torchvision on BML."
            ) from exc

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        model.eval()
        model.to(self._device)
        self._numpy = np
        self._person_detector = model
        return self._person_detector

    def _pil_to_tensor(self, image):
        rgb = self._numpy.asarray(image)
        return self._torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


class HogThenTorchvisionPersonCandidateGenerator:
    """Run HOG first, then use torchvision only when HOG finds no candidates."""

    def __init__(
        self,
        *,
        primary_generator: OpenCVHogPersonCandidateGenerator | object | None = None,
        fallback_generator: TorchvisionPersonCandidateGenerator | object | None = None,
        score_threshold: float = 0.3,
    ) -> None:
        self._primary_generator = (
            OpenCVHogPersonCandidateGenerator() if primary_generator is None else primary_generator
        )
        self._fallback_generator = (
            TorchvisionPersonCandidateGenerator(score_threshold=score_threshold)
            if fallback_generator is None
            else fallback_generator
        )

    def generate(self, sample: ConstructionSiteSample) -> tuple[PersonCandidate, ...]:
        """Return HOG candidates, or fall back to torchvision when HOG is empty."""
        primary_candidates = self._primary_generator.generate(sample)
        if primary_candidates:
            return primary_candidates
        return self._fallback_generator.generate(sample)


def _load_rgb_array(sample: ConstructionSiteSample):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime dependency only
        raise ImportError("Rule 1 person detection requires numpy and Pillow.") from exc

    return np.asarray(_load_pil_image(sample))


def _load_pil_image(sample: ConstructionSiteSample):
    if sample.image is None or sample.image.bytes is None:
        raise ValueError(f"Sample {sample.image_id} does not contain embedded image bytes.")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime dependency only
        raise ImportError("Rule 1 person detection requires Pillow.") from exc
    return Image.open(io.BytesIO(sample.image.bytes)).convert("RGB")


def _as_python_list(values: object) -> list[object]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        return list(values.tolist())
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return list(values)
