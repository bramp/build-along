"""Configuration for the progress bar indicator classifier."""

from pydantic import BaseModel


class ProgressBarIndicatorConfig(BaseModel):
    """Configuration for progress bar indicator classification.

    The progress bar indicator is a circular graphic on top of the progress bar
    that shows how far through the instructions the reader is. It's roughly
    square (circle) and typically 10-20 pixels in size.
    """

    min_size: float = 8.0
    """Minimum width/height for the indicator (filters out tiny elements)."""

    max_size: float = 25.0
    """Maximum width/height for the indicator (filters out large elements)."""

    max_aspect_ratio: float = 1.5
    """Maximum aspect ratio (width/height or height/width). 1.0 = perfect square."""

    bottom_margin_threshold: float = 0.25
    """Maximum distance from bottom of page as a ratio of page height."""
