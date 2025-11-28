from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, Field


class PageRange(BaseModel):
    """A 1-indexed inclusive page range selection.

    Either bound may be None to indicate from start or to end respectively.
    """

    model_config = ConfigDict(frozen=True)

    start: int | None
    end: int | None

    def __str__(self) -> str:
        # Single page
        if self.start is not None and self.end is not None and self.start == self.end:
            return str(self.start)
        # Open start
        if self.start is None and self.end is not None:
            return f"-{self.end}"
        # Open end
        if self.start is not None and self.end is None:
            return f"{self.start}-"
        # Explicit range (including potentially invalid where start>end, though parser
        # prevents it)
        if self.start is not None and self.end is not None:
            return f"{self.start}-{self.end}"
        # Fallback (shouldn't happen)
        return "-"


class PageRanges(BaseModel):
    """Collection of PageRange helpers.

    Responsible for converting user-specified 1-indexed ranges into concrete
    0-indexed page indices for a specific document, including clamping and
    de-duplication while preserving order.
    """

    model_config = ConfigDict(frozen=True)

    ranges: tuple[PageRange, ...] = Field(default_factory=tuple)

    def __str__(self) -> str:
        if not self.ranges:
            return "all"
        return ",".join(str(r) for r in self.ranges)

    @classmethod
    def all(cls) -> PageRanges:
        """Return a PageRanges instance representing all pages.

        Semantically equivalent to an empty ranges tuple in this design.
        """
        return cls()

    def page_numbers(self, num_pages: int) -> Iterator[int]:
        """Yield 1-indexed page numbers expanded from ranges, clamped and deduped.

        Args:
            num_pages: Total number of pages in the document.

        Yields:
            1-indexed page numbers in order with de-duplication.
        """
        if num_pages <= 0:
            return

        if not self.ranges:
            # Default: all pages (yield 1..num_pages)
            yield from range(1, num_pages + 1)
            return

        seen: set[int] = set()
        for rng in self.ranges:
            first = (rng.start - 1) if rng.start is not None else 0
            last = (rng.end - 1) if rng.end is not None else (num_pages - 1)
            # Clamp into valid bounds
            first = max(0, min(first, num_pages - 1))
            last = max(0, min(last, num_pages - 1))
            if first > last:
                continue
            for i in range(first, last + 1):
                if i not in seen:
                    seen.add(i)
                    yield i + 1


def parse_page_range(page_str: str) -> tuple[int | None, int | None]:
    """Parse a page range string into start and end page numbers.

    Supported formats:
    - "5": Single page (returns 5, 5)
    - "5-10": Page range from 5 to 10 (returns 5, 10)
    - "10-": From page 10 to end (returns 10, None)
    - "-5": From start to page 5 (returns None, 5)

    Args:
        page_str: The page range string to parse.

    Returns:
        A tuple of (start_page, end_page), where None indicates an unbounded range.

    Raises:
        ValueError: If the page range format is invalid or contains invalid numbers.
    """
    page_str = page_str.strip()
    if not page_str:
        raise ValueError("Page range cannot be empty")

    # Check for range syntax
    if "-" in page_str:
        parts = page_str.split("-", 1)
        start_str = parts[0].strip()
        end_str = parts[1].strip()

        # Handle "-5" format (start to page 5)
        if not start_str:
            if not end_str:
                raise ValueError(
                    "Invalid page range: '-'. At least one page number required."
                )
            try:
                end_page = int(end_str)
                # Reject negative numbers - they look like "-5" but are actually
                # negative
                if end_str.startswith("-"):
                    raise ValueError(f"Page number must be >= 1, got {end_page}")
                if end_page < 1:
                    raise ValueError(f"Page number must be >= 1, got {end_page}")
                return None, end_page
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid end page number: '{end_str}'") from None
                raise

        # Handle "10-" format (page 10 to end)
        if not end_str:
            try:
                start_page = int(start_str)
                if start_page < 1:
                    raise ValueError(f"Page number must be >= 1, got {start_page}")
                return start_page, None
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(
                        f"Invalid start page number: '{start_str}'"
                    ) from None
                raise

        # Handle "5-10" format (explicit range)
        try:
            start_page = int(start_str)
        except ValueError:
            raise ValueError(f"Invalid start page number: '{start_str}'") from None

        try:
            end_page = int(end_str)
        except ValueError:
            raise ValueError(f"Invalid end page number: '{end_str}'") from None

        if start_page < 1:
            raise ValueError(f"Start page must be >= 1, got {start_page}")
        if end_page < 1:
            raise ValueError(f"End page must be >= 1, got {end_page}")
        if start_page > end_page:
            raise ValueError(
                f"Start page ({start_page}) cannot be greater than end page "
                f"({end_page})"
            )
        return start_page, end_page
    else:
        # Single page number
        try:
            page_num = int(page_str)
            if page_num < 1:
                raise ValueError(f"Page number must be >= 1, got {page_num}")
            return page_num, page_num
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid page number: '{page_str}'") from None
            raise


def parse_page_ranges(pages_str: str) -> PageRanges:
    """Parse a comma-separated set of page segments into PageRanges."""
    if pages_str.strip() == "all":
        return PageRanges.all()

    segments = [seg.strip() for seg in pages_str.split(",") if seg.strip()]
    if not segments:
        raise ValueError("Invalid --pages value: empty after parsing")

    ranges: list[PageRange] = []
    for seg in segments:
        start, end = parse_page_range(seg)
        ranges.append(PageRange(start=start, end=end))
    return PageRanges(ranges=tuple(ranges))
