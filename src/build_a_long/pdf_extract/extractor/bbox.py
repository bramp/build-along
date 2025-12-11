from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Protocol

from annotated_types import Ge
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from build_a_long.pdf_extract.extractor.pymupdf_types import RectLike

# Type alias for non-negative floats
NonNegativeFloat = Annotated[float, Ge(0)]


class BBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    def __init__(
        self,
        x0: float | None = None,
        y0: float | None = None,
        x1: float | None = None,
        y1: float | None = None,
        /,
        **kwargs,
    ):
        """Initialize BBox with positional or keyword arguments.

        Supports both:
        - BBox(0, 0, 10, 10)  # positional
        - BBox(x0=0, y0=0, x1=10, y1=10)  # keyword
        """
        # If positional args are provided, use them
        if x0 is not None and y0 is not None and x1 is not None and y1 is not None:
            super().__init__(x0=x0, y0=y0, x1=x1, y1=y1, **kwargs)
        else:
            # Otherwise use keyword args
            super().__init__(**kwargs)

    def model_post_init(self, __context) -> None:
        """Validate x0 <= x1 and y0 <= y1."""
        if self.x0 > self.x1:
            raise ValueError(f"x0 ({self.x0}) must not be greater than x1 ({self.x1})")
        if self.y0 > self.y1:
            raise ValueError(f"y0 ({self.y0}) must not be greater than y1 ({self.y1})")

    def __str__(self) -> str:
        """Return a compact string representation of the bounding box."""
        return f"({self.x0:.1f},{self.y0:.1f},{self.x1:.1f},{self.y1:.1f})"

    @classmethod
    def from_tuple(cls, bbox_tuple: tuple[float, float, float, float]) -> BBox:
        """Create a BBox from a tuple of four floats (x0, y0, x1, y1)."""
        return cls(
            x0=bbox_tuple[0],
            y0=bbox_tuple[1],
            x1=bbox_tuple[2],
            y1=bbox_tuple[3],
        )

    @classmethod
    def from_rect(cls, rect: RectLike) -> BBox:
        """Create a BBox from a PyMuPDF Rect-like object.

        Args:
            rect: Any object with x0, y0, x1, y1 attributes (e.g., pymupdf.Rect)

        Returns:
            A new BBox with the coordinates from the rect
        """
        return cls(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1)

    def equals(self, other: BBox) -> bool:
        """
        Checks if this bounding box is equal to another bounding box.
        """
        return (
            self.x0 == other.x0
            and self.y0 == other.y0
            and self.x1 == other.x1
            and self.y1 == other.y1
        )

    def similar(self, other: BBox, tolerance: float = 1.0) -> bool:
        """
        Checks if this bounding box is nearly identical to another within tolerance.

        Args:
            other: The other bounding box to compare.
            tolerance: Maximum difference allowed for each coordinate.

        Returns:
            True if all coordinates differ by at most the tolerance.
        """
        return (
            abs(self.x0 - other.x0) <= tolerance
            and abs(self.y0 - other.y0) <= tolerance
            and abs(self.x1 - other.x1) <= tolerance
            and abs(self.y1 - other.y1) <= tolerance
        )

    def overlaps(self, other: BBox) -> bool:
        """
        Checks if this bounding box overlaps with another bounding box.
        """
        # If one rectangle is to the right of the other
        if self.x0 >= other.x1 or other.x0 >= self.x1:
            return False
        # If one rectangle is above the other
        return not (self.y0 >= other.y1 or other.y0 >= self.y1)

    def contains(self, other: BBox) -> bool:
        """
        Checks if this bounding box fully contains another bounding box.
        This is the inverse of fully_inside - returns True if other is inside self.
        """
        return (
            other.x0 >= self.x0
            and other.y0 >= self.y0
            and other.x1 <= self.x1
            and other.y1 <= self.y1
        )

    def adjacent(self, other: BBox, tolerance: float = 1e-6) -> bool:
        """
        Checks if this bounding box is adjacent to another bounding box
        (they are touching).
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

    def overlaps_horizontal(self, other: BBox) -> bool:
        """Helper to check if horizontal projections overlap."""
        return max(self.x0, other.x0) < min(self.x1, other.x1)

    def overlaps_vertical(self, other: BBox) -> bool:
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

    def intersection_area(self, other: BBox) -> float:
        """Return the area of intersection between this bbox and another."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        w = max(0.0, ix1 - ix0)
        h = max(0.0, iy1 - iy0)
        return w * h

    def intersect(self, other: BBox) -> BBox:
        """Return the intersection bbox between this bbox and another.

        If there is no intersection, returns a zero-area bbox at the
        closest point of approach.
        """
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)

        # Ensure valid bbox (x0 <= x1, y0 <= y1)
        if ix0 > ix1:
            ix0 = ix1 = (ix0 + ix1) / 2
        if iy0 > iy1:
            iy0 = iy1 = (iy0 + iy1) / 2

        return BBox(x0=ix0, y0=iy0, x1=ix1, y1=iy1)

    def iou(self, other: BBox) -> float:
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

    def min_distance(self, other: BBox) -> float:
        """Calculate minimum distance between this bbox and another.

        Returns 0.0 if the bboxes overlap or touch.
        Otherwise returns the minimum Euclidean distance between any two points
        on the bbox edges.

        Args:
            other: The other BBox to measure distance to.

        Returns:
            Minimum distance between the bboxes (0.0 if overlapping).
        """
        # If they overlap, distance is 0
        if self.overlaps(other):
            return 0.0

        # Calculate horizontal distance
        if self.x1 < other.x0:
            dx = other.x0 - self.x1
        elif other.x1 < self.x0:
            dx = self.x0 - other.x1
        else:
            dx = 0.0

        # Calculate vertical distance
        if self.y1 < other.y0:
            dy = other.y0 - self.y1
        elif other.y1 < self.y0:
            dy = self.y0 - other.y1
        else:
            dy = 0.0

        # Return Euclidean distance
        return (dx**2 + dy**2) ** 0.5

    @property
    def center(self) -> tuple[float, float]:
        """Return the (x, y) center point of the bbox."""
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Convert BBox to tuple (x0, y0, x1, y1).

        Useful for interfacing with PIL and other libraries that expect tuples.
        """
        return (self.x0, self.y0, self.x1, self.y1)

    def union(self, other: BBox) -> BBox:
        """Return the bounding box that encompasses both this bbox and another.

        Args:
            other: The other BBox to union with.

        Returns:
            A new BBox that contains both bounding boxes.
        """
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    @classmethod
    def union_all(cls, bboxes: list[BBox]) -> BBox:
        """Return the bounding box that encompasses all provided bboxes.

        Args:
            bboxes: List of BBox objects to union. Must be non-empty.

        Returns:
            A new BBox that contains all bounding boxes.

        Raises:
            ValueError: If bboxes list is empty.
        """
        if not bboxes:
            raise ValueError("Cannot compute union of empty list of bboxes")

        if len(bboxes) == 1:
            return bboxes[0]

        return BBox(
            x0=min(b.x0 for b in bboxes),
            y0=min(b.y0 for b in bboxes),
            x1=max(b.x1 for b in bboxes),
            y1=max(b.y1 for b in bboxes),
        )

    def clip_to(self, bounds: BBox) -> BBox:
        """Clip this bounding box to stay within the given bounds.

        Args:
            bounds: The bounding box to clip to.

        Returns:
            A new BBox clipped to the bounds. If this bbox doesn't overlap
            with bounds at all, returns a degenerate bbox (x0 == x1 or y0 == y1)
            at the nearest edge.
        """
        x0 = max(self.x0, bounds.x0)
        y0 = max(self.y0, bounds.y0)
        x1 = min(self.x1, bounds.x1)
        y1 = min(self.y1, bounds.y1)

        # If no overlap, clamp to create a degenerate (zero-area) bbox
        if x0 > x1:
            x0 = x1 = max(bounds.x0, min(self.x0, bounds.x1))
        if y0 > y1:
            y0 = y1 = max(bounds.y0, min(self.y0, bounds.y1))

        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def expand(self, margin: float) -> BBox:
        """Return a new BBox expanded by the given margin on all sides.

        Args:
            margin: The amount to expand the bbox by. Can be negative to shrink.

        Returns:
            A new BBox expanded by the margin.

        Raises:
            ValueError: If a negative margin would result in an invalid bbox
                (width or height < 0).
        """
        x0 = self.x0 - margin
        y0 = self.y0 - margin
        x1 = self.x1 + margin
        y1 = self.y1 + margin

        if x0 > x1 or y0 > y1:
            raise ValueError(
                f"Cannot expand bbox by {margin}: result would be invalid "
                f"(width={x1 - x0}, height={y1 - y0})"
            )

        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


class HasBBox(Protocol):
    """Protocol for objects that have a bbox attribute."""

    @property
    def bbox(self) -> BBox: ...


def build_connected_cluster[T: HasBBox](
    seed_item: T,
    candidate_items: list[T],
) -> list[T]:
    """Build a connected cluster of items based on bbox overlap.

    Starts with a seed item and recursively adds candidates that overlap
    with any item already in the cluster.

    Args:
        seed_item: Initial item to start the cluster
        candidate_items: Items to consider adding to the cluster

    Returns:
        List of items in the connected cluster (includes seed item)

    Example:
        >>> # Find images that form a connected cluster with a bag number
        >>> cluster = build_connected_cluster(bag_image, images)
    """
    # Build index mapping for quick lookup
    candidate_set = set(range(len(candidate_items)))
    cluster_indices: set[int] = set()
    to_process: list[int] = []

    # Add seed item to cluster
    for idx, candidate in enumerate(candidate_items):
        if candidate is seed_item or candidate.bbox.equals(seed_item.bbox):
            if idx in candidate_set:
                cluster_indices.add(idx)
                to_process.append(idx)
                candidate_set.discard(idx)
            break

    # Expand cluster by finding overlapping items
    processed: set[int] = set()
    while to_process:
        current_idx = to_process.pop()
        if current_idx in processed:
            continue
        processed.add(current_idx)

        current_item = candidate_items[current_idx]

        # Find candidates that overlap with current item
        for idx in list(candidate_set):
            candidate = candidate_items[idx]
            if candidate.bbox.overlaps(current_item.bbox):
                cluster_indices.add(idx)
                to_process.append(idx)
                candidate_set.discard(idx)

    # Return clustered items in original order
    return [candidate_items[idx] for idx in sorted(cluster_indices)]


def build_all_connected_clusters[T: HasBBox](items: list[T]) -> list[list[T]]:
    """Build all connected clusters from a list of items based on bbox overlap.

    Groups all items into clusters where items in each cluster are
    transitively connected through overlapping bounding boxes.

    Args:
        items: List of items with bbox property

    Returns:
        List of clusters, where each cluster is a list of connected items

    Example:
        >>> # Find all groups of overlapping images on a page
        >>> clusters = build_all_connected_clusters(images)
        >>> for cluster in clusters:
        ...     print(f"Cluster of {len(cluster)} images")
    """
    if not items:
        return []

    # Track which items have been assigned to clusters
    remaining = set(range(len(items)))
    clusters: list[list[T]] = []

    while remaining:
        # Pick an arbitrary seed from remaining items
        seed_idx = min(remaining)
        seed_item = items[seed_idx]
        remaining.remove(seed_idx)

        # Build a cluster starting from this seed
        cluster = build_connected_cluster(seed_item, items)

        # Remove clustered items from remaining set
        for item in cluster:
            try:
                idx = items.index(item)
                remaining.discard(idx)
            except ValueError:
                pass

        clusters.append(cluster)

    return clusters


def filter_contained[T: HasBBox](items: list[T], container: BBox) -> list[T]:
    """Filter items to keep only those fully contained within the container bbox.

    Args:
        items: List of items with bbox property
        container: The bounding box to check containment against

    Returns:
        List of items fully contained in the container
    """
    return [item for item in items if container.contains(item.bbox)]


def filter_overlapping[T: HasBBox](items: list[T], target: BBox) -> list[T]:
    """Filter items to keep only those overlapping with the target bbox.

    Args:
        items: List of items with bbox property
        target: The bounding box to check overlap against

    Returns:
        List of items overlapping with the target
    """
    return [item for item in items if target.overlaps(item.bbox)]


def filter_by_max_area[T: HasBBox](
    items: list[T],
    max_area: float | None = None,
    max_ratio: float | None = None,
    reference_bbox: BBox | None = None,
) -> list[T]:
    """Filter items to exclude those exceeding a maximum area.

    Useful for excluding full-page backgrounds or other large elements
    when looking for smaller UI components.

    Must specify either max_area OR (max_ratio AND reference_bbox).

    Args:
        items: List of items with bbox property
        max_area: Maximum allowed area in absolute units (e.g., square points)
        max_ratio: Maximum allowed area as a ratio of reference_bbox area
            (e.g., 0.5 for 50% of page size)
        reference_bbox: Reference bbox for ratio calculation (typically page bbox)

    Returns:
        List of items with area <= threshold

    Raises:
        ValueError: If neither max_area nor (max_ratio, reference_bbox) provided

    Example:
        >>> # Filter out drawings larger than 50% of the page
        >>> small_drawings = filter_by_max_area(
        ...     drawings, max_ratio=0.5, reference_bbox=page_bbox
        ... )
    """
    if max_area is not None:
        threshold = max_area
    elif max_ratio is not None and reference_bbox is not None:
        threshold = reference_bbox.area * max_ratio
    else:
        raise ValueError(
            "Must specify either max_area or (max_ratio and reference_bbox)"
        )

    return [item for item in items if item.bbox.area <= threshold]


def group_by_similar_bbox[T: HasBBox](
    items: list[T],
    tolerance: float = 2.0,
) -> list[list[T]]:
    """Group items by similar bounding boxes.

    Items with nearly identical bboxes (within tolerance) are grouped together.
    This is useful for grouping duplicate elements like overlapping drawing
    borders or fill regions that represent the same logical element.

    Uses greedy grouping: each item is added to the first group with a similar
    bbox, or starts a new group if no match is found. Similarity is checked
    against the first item in each group.

    Args:
        items: List of items with bbox property
        tolerance: Maximum coordinate difference to consider bboxes similar
            (default 2.0 points)

    Returns:
        List of groups, where each group contains items with similar bboxes.
        Groups preserve insertion order.

    Example:
        >>> # Group drawings that represent the same parts list border
        >>> groups = group_by_similar_bbox(drawings, tolerance=2.0)
        >>> for group in groups:
        ...     bbox = BBox.union_all([d.bbox for d in group])
        ...     print(f"Group of {len(group)} drawings at {bbox}")
    """
    if not items:
        return []

    groups: list[list[T]] = []
    for item in items:
        # Try to find an existing group with similar bbox
        found_group = False
        for group in groups:
            if item.bbox.similar(group[0].bbox, tolerance=tolerance):
                group.append(item)
                found_group = True
                break
        if not found_group:
            groups.append([item])

    return groups
