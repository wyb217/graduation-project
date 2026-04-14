"""Tests for Rule 1 person candidate utilities."""

from __future__ import annotations

import pytest

from point1.candidates.person import pixel_xyxy_to_normalized_bbox


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
