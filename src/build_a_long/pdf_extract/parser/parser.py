from typing import Tuple


def parse_page_range(page_str: str) -> Tuple[int | None, int | None]:
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
    # TODO in future support accepting lists, e.g "1, 2, 3" or "1-3,5,7-9"

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
                # Reject negative numbers - they look like "-5" but are actually negative
                if end_str.startswith("-"):
                    raise ValueError(f"Page number must be >= 1, got {end_page}")
                if end_page < 1:
                    raise ValueError(f"Page number must be >= 1, got {end_page}")
                return None, end_page
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid end page number: '{end_str}'")
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
                    raise ValueError(f"Invalid start page number: '{start_str}'")
                raise

        # Handle "5-10" format (explicit range)
        try:
            start_page = int(start_str)
        except ValueError:
            raise ValueError(f"Invalid start page number: '{start_str}'")

        try:
            end_page = int(end_str)
        except ValueError:
            raise ValueError(f"Invalid end page number: '{end_str}'")

        if start_page < 1:
            raise ValueError(f"Start page must be >= 1, got {start_page}")
        if end_page < 1:
            raise ValueError(f"End page must be >= 1, got {end_page}")
        if start_page > end_page:
            raise ValueError(
                f"Start page ({start_page}) cannot be greater than end page ({end_page})"
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
                raise ValueError(f"Invalid page number: '{page_str}'")
            raise
