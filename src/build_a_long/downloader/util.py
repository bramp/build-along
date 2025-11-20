"""General utility functions for the downloader."""

from pathlib import PurePosixPath
from urllib.parse import urlparse

from pydantic import AnyUrl


def is_valid_set_id(set_id: str) -> bool:
    """
    Validates if a string is a valid LEGO set ID (only digits).
    """
    return set_id.isdigit()


def extract_filename_from_url(url: AnyUrl | str) -> str | None:
    """Extract filename from a URL, handling edge cases.

    Args:
        url: URL to extract filename from (can be AnyUrl or str)

    Returns:
        Extracted filename, or None if ambiguous/not found

    Examples:
        >>> extract_filename_from_url("https://example.com/file.pdf")
        'file.pdf'
        >>> extract_filename_from_url("https://example.com/path/to/file.pdf")
        'file.pdf'
        >>> extract_filename_from_url("https://example.com/")
        None
        >>> extract_filename_from_url("https://example.com")
        None
    """
    # Get the path component from the URL
    if isinstance(url, AnyUrl):
        path = url.path or ""
    else:
        # For string URLs, parse and extract path
        parsed = urlparse(str(url))
        path = parsed.path or ""

    # If no path or root path only, return None
    if not path or path == "/":
        return None

    # URLs ending with / are directories, not files
    if path.endswith("/"):
        return None

    # Use PurePosixPath to extract the filename (works for URL paths)
    filename = PurePosixPath(path).name

    # If no filename found, return None
    if not filename:
        return None

    return filename
