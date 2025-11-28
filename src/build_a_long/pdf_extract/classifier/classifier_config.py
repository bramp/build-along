"""Configuration for the classifier."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.pages.page_hint_collection import (
    PageHintCollection,
)
from build_a_long.pdf_extract.classifier.score import Weight
from build_a_long.pdf_extract.classifier.text import FontSizeHints


class ClassifierConfig(BaseModel):
    """Configuration for the classifier.

    Naming Conventions
    ------------------
    All classifier-specific settings should be prefixed with the label name:

    - `{label}_min_score`: Minimum score threshold. Candidates scoring below
      this value are not created (to reduce debug spam). Default: 0.5
    - `{label}_*_weight`: Weights for different scoring components
    - `{label}_*`: Other label-specific configuration

    Example: For a "page_number" label:
    - page_number_min_score
    - page_number_text_weight
    - page_number_position_weight
    """

    # TODO Consistenctly use this, or give it a name more descriptive of where
    # it's used
    min_confidence_threshold: Weight = 0.6

    # Page number classifier settings
    page_number_min_score: Weight = 0.5
    page_number_text_weight: Weight = 0.7
    page_number_position_weight: Weight = 0.3
    page_number_position_scale: float = 50.0
    page_number_page_value_weight: Weight = 1.0
    page_number_font_size_weight: Weight = 0.1

    # Step number classifier settings
    step_number_min_score: Weight = 0.5
    step_number_text_weight: Weight = 0.7
    step_number_font_size_weight: Weight = 0.3

    # Part count classifier settings
    part_count_min_score: Weight = 0.5
    part_count_text_weight: Weight = 0.7
    part_count_font_size_weight: Weight = 0.3

    # Part number classifier settings
    part_number_min_score: Weight = 0.5

    # Parts list classifier settings
    parts_list_min_score: Weight = 0.5
    parts_list_max_area_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    """Maximum ratio of page area a parts list can occupy (0.0-1.0).
    
    Drawings larger than this fraction of the page are rejected as they're
    likely the entire page background rather than actual parts lists.
    Default is 0.75 (75% of page area).
    """

    # Rotation symbol classifier settings
    rotation_symbol_min_score: Weight = 0.3
    """Minimum score threshold for rotation symbol candidates."""

    rotation_symbol_ideal_size: float = 46.0
    """Ideal width/height in points for rotation symbols (used for scoring)."""

    rotation_symbol_min_size: float = rotation_symbol_ideal_size * 0.90
    """Minimum width/height in points for rotation symbols."""

    rotation_symbol_max_size: float = rotation_symbol_ideal_size * 1.10
    """Maximum width/height in points for rotation symbols."""

    rotation_symbol_min_aspect: float = 0.95
    """Minimum aspect ratio (width/height) - must be close to square."""

    rotation_symbol_max_aspect: float = 1.05
    """Maximum aspect ratio (width/height) - must be close to square."""

    rotation_symbol_min_drawings_in_cluster: int = 4
    """Minimum number of drawings to form a rotation symbol cluster."""

    rotation_symbol_max_drawings_in_cluster: int = 25
    """Maximum number of drawings to form a rotation symbol cluster."""

    rotation_symbol_size_weight: Weight = 0.5
    """Weight for size score in final rotation symbol score."""

    rotation_symbol_aspect_weight: Weight = 0.3
    """Weight for aspect ratio score in final rotation symbol score."""

    rotation_symbol_proximity_weight: Weight = 0.2
    """Weight for proximity to diagram score in final rotation symbol score."""

    rotation_symbol_cluster_size_score: float = 0.7
    """Base size score for drawing clusters (slightly lower than images)."""

    rotation_symbol_proximity_close_distance: float = 100.0
    """Distance in points within which proximity score is 1.0."""

    rotation_symbol_proximity_far_distance: float = 300.0
    """Distance in points beyond which proximity score is 0.0."""

    # NewBag classifier settings (for numberless bag detection)
    new_bag_min_score: Weight = 0.5
    """Minimum score threshold for new bag candidates."""

    new_bag_icon_min_size: float = 180.0
    """Minimum width/height in points for bag icon images.
    
    Based on observed data: bag icons are ~240x240 with overlapping images
    as small as 165x177. Using 180 to capture most bag icon components.
    """

    new_bag_icon_min_aspect: float = 0.90
    """Minimum aspect ratio (width/height) for bag icon images."""

    new_bag_icon_max_aspect: float = 1.10
    """Maximum aspect ratio (width/height) for bag icon images."""

    new_bag_icon_max_x_ratio: float = 0.15
    """Maximum x position as ratio of page width.
    
    Bag icons are typically positioned very close to the top-left corner,
    around x=14 on a 552pt wide page (~2.5%). Using 15% to allow some margin.
    """

    new_bag_icon_max_y_ratio: float = 0.15
    """Maximum y position as ratio of page height.
    
    Bag icons are typically positioned very close to the top-left corner,
    around y=14 on a 496pt tall page (~2.8%). Using 15% to allow some margin.
    """

    font_size_hints: FontSizeHints = Field(default_factory=FontSizeHints.empty)
    """Font size hints derived from analyzing all pages"""

    page_hints: PageHintCollection = Field(default_factory=PageHintCollection.empty)
    """Page type hints derived from analyzing all pages"""
