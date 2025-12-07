"""Configuration for divider classification."""

from __future__ import annotations

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.score import Weight


class DividerConfig(BaseModel):
    """Configuration for divider classification.

    Dividers are thin lines that separate sections of a page. They run
    vertically or horizontally across a significant portion of the page.
    """

    min_score: Weight = 0.5
    """Minimum score threshold for divider candidates."""

    min_length_ratio: float = 0.4
    """Minimum length as a ratio of page height (vertical) or width (horizontal).
    
    A divider must span at least this much of the page to be considered valid.
    Default is 0.4 (40% of page dimension).
    """

    max_thickness: float = 5.0
    """Maximum thickness in points for the divider line.
    
    Dividers are thin lines, so this limits how thick they can be.
    A value of 0 indicates a stroke-only line (no fill width).
    """

    # TODO Use hints from the page to determine the margin.
    edge_margin: float = 5.0
    """Margin in points from page edge to ignore dividers.
    
    Dividers within this distance from the page boundary are considered
    page borders and are filtered out. This prevents detecting page
    border lines as content dividers.
    """
