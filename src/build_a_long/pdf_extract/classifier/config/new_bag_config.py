"""Configuration for new bag classification."""

from __future__ import annotations

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.score import Weight


class NewBagConfig(BaseModel):
    """Configuration for new bag classification.

    NewBag elements mark when a new numbered bag of pieces should be opened.
    They consist of an optional bag number surrounded by a cluster of images
    forming a bag icon graphic, typically at the top-left of a page.
    """

    min_score: Weight = 0.5
    """Minimum score threshold for new bag candidates."""

    icon_min_size: float = 180.0
    """Minimum width/height in points for bag icon images.
    
    Based on observed data: bag icons are ~240x240 with overlapping images
    as small as 165x177. Using 180 to capture most bag icon components.
    """

    icon_min_aspect: float = 0.90
    """Minimum aspect ratio (width/height) for bag icon images."""

    icon_max_aspect: float = 1.10
    """Maximum aspect ratio (width/height) for bag icon images."""

    icon_max_x_ratio: float = 0.15
    """Maximum x position as ratio of page width.
    
    Bag icons are typically positioned very close to the top-left corner,
    around x=14 on a 552pt wide page (~2.5%). Using 15% to allow some margin.
    """

    icon_max_y_ratio: float = 0.15
    """Maximum y position as ratio of page height.
    
    Bag icons are typically positioned very close to the top-left corner,
    around y=14 on a 496pt tall page (~2.8%). Using 15% to allow some margin.
    """
