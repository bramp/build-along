"""Configuration for the classifier."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.font_size_hints import FontSizeHints
from build_a_long.pdf_extract.classifier.pages.page_hint_collection import (
    PageHintCollection,
)
from build_a_long.pdf_extract.classifier.score import Weight


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

    font_size_hints: FontSizeHints = Field(default_factory=FontSizeHints.empty)
    """Font size hints derived from analyzing all pages"""

    page_hints: PageHintCollection = Field(default_factory=PageHintCollection.empty)
    """Page type hints derived from analyzing all pages"""
