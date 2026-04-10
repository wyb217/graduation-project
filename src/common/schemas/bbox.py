"""Normalized bounding box contracts used across the repository."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizedBBox:
    """A normalized `xyxy` bounding box with coordinates in the range [0.0, 1.0]."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name, value in (
            ("x_min", self.x_min),
            ("y_min", self.y_min),
            ("x_max", self.x_max),
            ("y_max", self.y_max),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}.")

        if self.x_min > self.x_max:
            raise ValueError("x_min must be less than or equal to x_max.")
        if self.y_min > self.y_max:
            raise ValueError("y_min must be less than or equal to y_max.")

    @classmethod
    def from_list(cls, values: list[float]) -> NormalizedBBox:
        """Build a bbox from a four-number list."""
        if len(values) != 4:
            raise ValueError(f"Expected four bbox values, got {len(values)}.")
        return cls(*[float(value) for value in values])

    def to_list(self) -> list[float]:
        """Return the bbox as a JSON-friendly list."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]
