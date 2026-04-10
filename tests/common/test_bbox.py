"""Tests for normalized bounding boxes."""

from __future__ import annotations

import pytest

from common.schemas.bbox import NormalizedBBox


def test_bbox_accepts_valid_xyxy() -> None:
    """A valid normalized xyxy box should serialize cleanly."""
    bbox = NormalizedBBox(0.1, 0.2, 0.8, 0.9)

    assert bbox.to_list() == [0.1, 0.2, 0.8, 0.9]


def test_bbox_rejects_invalid_coordinate_order() -> None:
    """The box must enforce x_min <= x_max and y_min <= y_max."""
    with pytest.raises(ValueError, match="x_min"):
        NormalizedBBox(0.8, 0.2, 0.1, 0.9)


def test_bbox_rejects_out_of_range_values() -> None:
    """Normalized coordinates must stay inside [0, 1]."""
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        NormalizedBBox(-0.1, 0.2, 0.1, 0.9)
