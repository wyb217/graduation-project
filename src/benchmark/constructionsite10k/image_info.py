"""Image metadata helpers for parquet-embedded benchmark samples."""

from __future__ import annotations

from benchmark.constructionsite10k.types import ConstructionSiteSample

JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def get_jpeg_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    """Parse JPEG dimensions without requiring Pillow."""
    if image_bytes[:2] != b"\xff\xd8":
        return None

    index = 2
    while index < len(image_bytes):
        while index < len(image_bytes) and image_bytes[index] == 0xFF:
            index += 1
        if index >= len(image_bytes):
            return None

        marker = image_bytes[index]
        index += 1
        if marker in {0xD8, 0xD9}:
            continue
        if index + 1 >= len(image_bytes):
            return None

        segment_length = int.from_bytes(image_bytes[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > len(image_bytes):
            return None
        if marker in JPEG_SOF_MARKERS:
            if index + 7 > len(image_bytes):
                return None
            height = int.from_bytes(image_bytes[index + 3 : index + 5], "big")
            width = int.from_bytes(image_bytes[index + 5 : index + 7], "big")
            return width, height
        index += segment_length
    return None


def sample_has_max_image_side(
    sample: ConstructionSiteSample,
    *,
    max_image_side: int,
) -> bool:
    """Return whether the sample image stays inside a maximum width/height."""
    if sample.image is None or sample.image.bytes is None:
        return False
    dimensions = get_jpeg_dimensions(sample.image.bytes)
    if dimensions is None:
        return False
    return max(dimensions) <= max_image_side
