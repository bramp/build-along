from dataclasses import dataclass
from typing import Annotated, Tuple

from annotated_types import Ge

# Type alias for non-negative floats
NonNegativeFloat = Annotated[float, Ge(0)]


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self):
        if self.x0 > self.x1:
            raise ValueError(f"x0 ({self.x0}) must not be greater than x1 ({self.x1})")
        if self.y0 > self.y1:
            raise ValueError(f"y0 ({self.y0}) must not be greater than y1 ({self.y1})")

    def __str__(self) -> str:
        """Return a compact string representation of the bounding box."""
        return f"({self.x0:.1f},{self.y0:.1f},{self.x1:.1f},{self.y1:.1f})"

    @classmethod
    def from_tuple(cls, bbox_tuple: Tuple[float, float, float, float]) -> "BBox":
        """Create a BBox from a tuple of four floats (x0, y0, x1, y1)."""
        return cls(
            x0=bbox_tuple[0],
            y0=bbox_tuple[1],
            x1=bbox_tuple[2],
            y1=bbox_tuple[3],
        )

    def equals(self, other: "BBox") -> bool:
        """
        Checks if this bounding box is equal to another bounding box.
        """
        return (
            self.x0 == other.x0
            and self.y0 == other.y0
            and self.x1 == other.x1
            and self.y1 == other.y1
        )

    def overlaps(self, other: "BBox") -> bool:
        """
        Checks if this bounding box overlaps with another bounding box.
        """
        # If one rectangle is to the right of the other
        if self.x0 >= other.x1 or other.x0 >= self.x1:
            return False
        # If one rectangle is above the other
        if self.y0 >= other.y1 or other.y0 >= self.y1:
            return False
        return True

    def fully_inside(self, other: "BBox") -> bool:
        """
        Checks if this bounding box is fully inside another bounding box.
        """
        return (
            self.x0 >= other.x0
            and self.y0 >= other.y0
            and self.x1 <= other.x1
            and self.y1 <= other.y1
        )

    def adjacent(self, other: "BBox", tolerance: float = 1e-6) -> bool:
        """
        Checks if this bounding box is adjacent to another bounding box (they are touching).
        A small tolerance is used for floating point comparisons.
        """
        # Check for horizontal adjacency
        horizontal_adjacent = (
            abs(self.x1 - other.x0) < tolerance and self.overlaps_vertical(other)
        ) or (abs(other.x1 - self.x0) < tolerance and self.overlaps_vertical(other))
        # Check for vertical adjacency
        vertical_adjacent = (
            abs(self.y1 - other.y0) < tolerance and self.overlaps_horizontal(other)
        ) or (abs(other.y1 - self.y0) < tolerance and self.overlaps_horizontal(other))

        return horizontal_adjacent or vertical_adjacent

    def overlaps_horizontal(self, other: "BBox") -> bool:
        """Helper to check if horizontal projections overlap."""
        return max(self.x0, other.x0) < min(self.x1, other.x1)

    def overlaps_vertical(self, other: "BBox") -> bool:
        """Helper to check if vertical projections overlap."""
        return max(self.y0, other.y0) < min(self.y1, other.y1)

    @property
    def width(self) -> NonNegativeFloat:
        """Return the width of this bounding box (non-negative)."""
        return self.x1 - self.x0

    @property
    def height(self) -> NonNegativeFloat:
        """Return the height of this bounding box (non-negative)."""
        return self.y1 - self.y0

    @property
    def area(self) -> NonNegativeFloat:
        """Return the area of this bounding box (non-negative)."""
        return self.width * self.height

    def intersection_area(self, other: "BBox") -> float:
        """Return the area of intersection between this bbox and another."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        w = max(0.0, ix1 - ix0)
        h = max(0.0, iy1 - iy0)
        return w * h

    def iou(self, other: "BBox") -> float:
        """Intersection over Union with another bbox.

        Returns 0.0 when there is no overlap or union is zero.
        """
        inter = self.intersection_area(other)
        if inter == 0.0:
            return 0.0
        ua = self.area + other.area - inter
        if ua <= 0.0:
            return 0.0
        return inter / ua

    @property
    def center(self) -> Tuple[float, float]:
        """Return the (x, y) center point of the bbox."""
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)
