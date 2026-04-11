"""Parsers for raw ConstructionSite10k annotation dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmark.constructionsite10k.types import (
    ConstructionSiteSample,
    RuleViolation,
    SampleAttributes,
    SampleImage,
)
from common.schemas.bbox import NormalizedBBox

RULE_FIELD_PREFIX = "rule_"
RULE_FIELD_SUFFIX = "_violation"
BASE_SAMPLE_FIELDS = {
    "image",
    "image_id",
    "image_caption",
    "illumination",
    "camera_distance",
    "view",
    "quality_of_info",
}


def parse_sample(
    raw: Mapping[str, Any],
    *,
    include_image_bytes: bool = True,
) -> ConstructionSiteSample:
    """Parse one raw benchmark annotation into a typed sample."""
    image_id = str(raw["image_id"])
    attributes = SampleAttributes(
        illumination=str(raw["illumination"]),
        camera_distance=str(raw["camera_distance"]),
        view=str(raw["view"]),
        quality_of_info=str(raw["quality_of_info"]),
    )

    violations = _parse_violations(raw, image_id=image_id)
    object_boxes = _parse_object_boxes(raw)

    return ConstructionSiteSample(
        image_id=image_id,
        image=_parse_image(raw.get("image"), include_image_bytes=include_image_bytes),
        image_caption=str(raw["image_caption"]),
        attributes=attributes,
        violations=violations,
        object_boxes=object_boxes,
    )


def _parse_image(raw_image: Any, *, include_image_bytes: bool) -> SampleImage | None:
    if raw_image is None:
        return None
    if not isinstance(raw_image, Mapping):
        raise ValueError(f"Expected image payload to be a mapping, got {type(raw_image)!r}.")

    raw_bytes = raw_image.get("bytes")
    image_bytes = bytes(raw_bytes) if include_image_bytes and raw_bytes is not None else None
    raw_path = raw_image.get("path")
    image_path = str(raw_path) if raw_path is not None else None
    return SampleImage(path=image_path, bytes=image_bytes)


def _parse_violations(raw: Mapping[str, Any], *, image_id: str) -> dict[int, RuleViolation | None]:
    violations: dict[int, RuleViolation | None] = {}

    for key, value in raw.items():
        if not key.startswith(RULE_FIELD_PREFIX) or not key.endswith(RULE_FIELD_SUFFIX):
            continue

        rule_id_text = key[len(RULE_FIELD_PREFIX) : -len(RULE_FIELD_SUFFIX)]
        rule_id = int(rule_id_text)
        if value is None:
            violations[rule_id] = None
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{image_id}: expected mapping for {key}, got {type(value)!r}.")

        bounding_box_values = value.get("bounding_box", [])
        if not isinstance(bounding_box_values, list):
            raise ValueError(f"{image_id}: expected list for {key}.bounding_box.")

        violations[rule_id] = RuleViolation(
            rule_id=rule_id,
            reason=str(value.get("reason", "")),
            bounding_boxes=tuple(
                _parse_bbox_list(bounding_box_values, image_id=image_id, field=key)
            ),
        )

    return dict(sorted(violations.items()))


def _parse_object_boxes(raw: Mapping[str, Any]) -> dict[str, tuple[NormalizedBBox, ...]]:
    object_boxes: dict[str, tuple[NormalizedBBox, ...]] = {}

    for key, value in raw.items():
        if key in BASE_SAMPLE_FIELDS or (
            key.startswith(RULE_FIELD_PREFIX) and key.endswith(RULE_FIELD_SUFFIX)
        ):
            continue
        if not isinstance(value, list):
            continue
        object_boxes[key] = tuple(_parse_bbox_list(value, image_id=str(raw["image_id"]), field=key))

    return object_boxes


def _parse_bbox_list(
    raw_boxes: list[Any],
    *,
    image_id: str,
    field: str,
) -> list[NormalizedBBox]:
    boxes: list[NormalizedBBox] = []
    for index, raw_box in enumerate(raw_boxes):
        if not isinstance(raw_box, list):
            raise ValueError(f"{image_id}: {field}[{index}] must be a list of four numbers.")
        try:
            boxes.append(NormalizedBBox.from_list([float(value) for value in raw_box]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{image_id}: invalid bbox in {field}[{index}].") from exc
    return boxes
