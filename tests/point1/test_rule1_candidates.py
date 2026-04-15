"""Tests for Rule 1 person candidate utilities."""

from __future__ import annotations

import io

import pytest

from common.schemas.bbox import NormalizedBBox
from point1.candidates.person import (
    HogThenTorchvisionPersonCandidateGenerator,
    PersonCandidate,
    TorchvisionPersonCandidateGenerator,
    pixel_xyxy_to_normalized_bbox,
)


def test_pixel_xyxy_to_normalized_bbox_converts_coordinates() -> None:
    """Pixel-space person detections should map to normalized xyxy boxes."""
    bbox = pixel_xyxy_to_normalized_bbox(
        x_min=10,
        y_min=20,
        x_max=110,
        y_max=220,
        image_width=200,
        image_height=400,
    )

    assert bbox.to_list() == [0.05, 0.05, 0.55, 0.55]


def test_pixel_xyxy_to_normalized_bbox_rejects_non_positive_image_size() -> None:
    """Image dimensions must be positive when normalizing detector outputs."""
    with pytest.raises(ValueError, match="image_width and image_height must be positive"):
        pixel_xyxy_to_normalized_bbox(
            x_min=0,
            y_min=0,
            x_max=10,
            y_max=10,
            image_width=0,
            image_height=400,
        )


def _valid_png_bytes() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (200, 400), color=(240, 200, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_torchvision_person_candidate_generator_filters_to_people_and_min_size(
    sample_annotation: dict[str, object],
) -> None:
    """Torchvision fallback should keep only person detections above score/size thresholds."""
    from benchmark.constructionsite10k.parser import parse_sample

    sample = parse_sample(
        {
            **sample_annotation,
            "image": {"bytes": _valid_png_bytes(), "path": "demo.png"},
        }
    )

    class FakeDetector:
        def __call__(self, images):  # noqa: ANN001
            assert len(images) == 1
            return [
                {
                    "boxes": [
                        [10.0, 20.0, 90.0, 220.0],  # valid person
                        [5.0, 5.0, 20.0, 40.0],  # too small
                        [30.0, 40.0, 130.0, 260.0],  # wrong label
                        [40.0, 50.0, 120.0, 240.0],  # low score
                    ],
                    "labels": [1, 1, 3, 1],
                    "scores": [0.95, 0.99, 0.97, 0.1],
                }
            ]

    generator = TorchvisionPersonCandidateGenerator(
        person_detector=FakeDetector(),
        score_threshold=0.3,
        min_width=24,
        min_height=48,
    )

    candidates = generator.generate(sample)

    assert candidates == (
        PersonCandidate(
            candidate_id="person-1",
            bbox=NormalizedBBox(0.05, 0.05, 0.45, 0.55),
            score=0.95,
        ),
    )


def test_hog_then_torchvision_generator_skips_fallback_when_hog_finds_candidates() -> None:
    """Hybrid detector should preserve the HOG path when it already finds a person."""

    class FakePrimary:
        def generate(self, sample):  # noqa: ANN001
            return (PersonCandidate("person-1", NormalizedBBox(0.1, 0.2, 0.3, 0.6), 0.9),)

    class FakeFallback:
        def __init__(self) -> None:
            self.called = False

        def generate(self, sample):  # noqa: ANN001
            self.called = True
            return ()

    fallback = FakeFallback()
    generator = HogThenTorchvisionPersonCandidateGenerator(
        primary_generator=FakePrimary(),
        fallback_generator=fallback,
    )

    candidates = generator.generate(sample=None)

    assert len(candidates) == 1
    assert fallback.called is False


def test_hog_then_torchvision_generator_uses_fallback_when_hog_finds_nothing() -> None:
    """Hybrid detector should fall back to torchvision when HOG returns zero candidates."""

    class FakePrimary:
        def generate(self, sample):  # noqa: ANN001
            return ()

    class FakeFallback:
        def __init__(self) -> None:
            self.called = False

        def generate(self, sample):  # noqa: ANN001
            self.called = True
            return (PersonCandidate("person-2", NormalizedBBox(0.2, 0.2, 0.4, 0.7), 0.8),)

    fallback = FakeFallback()
    generator = HogThenTorchvisionPersonCandidateGenerator(
        primary_generator=FakePrimary(),
        fallback_generator=fallback,
    )

    candidates = generator.generate(sample=None)

    assert len(candidates) == 1
    assert candidates[0].candidate_id == "person-2"
    assert fallback.called is True


def test_torchvision_pil_to_tensor_copies_array_before_from_numpy() -> None:
    """Torchvision fallback should hand PyTorch a writable array copy."""
    from PIL import Image

    class FakeTensor:
        def permute(self, *args):  # noqa: ANN002
            return self

        def float(self):
            return self

        def __truediv__(self, other: float):  # noqa: ANN001
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.array_is_writable: bool | None = None

        def from_numpy(self, array):  # noqa: ANN001
            self.array_is_writable = bool(array.flags.writeable)
            return FakeTensor()

    generator = TorchvisionPersonCandidateGenerator(person_detector=object())
    generator._numpy = __import__("numpy")  # noqa: SLF001
    fake_torch = FakeTorch()
    generator._torch = fake_torch  # noqa: SLF001

    image = Image.new("RGB", (16, 16), color=(128, 128, 128))

    generator._pil_to_tensor(image)  # noqa: SLF001

    assert fake_torch.array_is_writable is True
