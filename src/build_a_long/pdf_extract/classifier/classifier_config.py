"""Configuration for the classifier."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.config import (
    ArrowConfig,
    BackgroundConfig,
    BagNumberConfig,
    DividerConfig,
    NewBagConfig,
    PageNumberConfig,
    PartCountConfig,
    PartNumberConfig,
    PartsListConfig,
    RotationSymbolConfig,
    StepCountConfig,
    StepNumberConfig,
    SubAssemblyConfig,
    TriviaTextConfig,
)
from build_a_long.pdf_extract.classifier.pages.page_hint_collection import (
    PageHintCollection,
)
from build_a_long.pdf_extract.classifier.score import Weight
from build_a_long.pdf_extract.classifier.text import FontSizeHints


class ClassifierConfig(BaseModel):
    """Configuration for the classifier."""

    # TODO Consistently use this, or give it a name more descriptive of where
    # it's used
    min_confidence_threshold: Weight = 0.6

    # Page number classifier settings
    page_number: PageNumberConfig = Field(default_factory=PageNumberConfig)
    """Configuration for page number classification."""

    # Step number classifier settings
    step_number: StepNumberConfig = Field(default_factory=StepNumberConfig)
    """Configuration for step number classification."""

    # Part count classifier settings
    part_count: PartCountConfig = Field(default_factory=PartCountConfig)
    """Configuration for part count classification."""

    # Part number classifier settings
    part_number: PartNumberConfig = Field(default_factory=PartNumberConfig)
    """Configuration for part number classification."""

    # Parts list classifier settings
    parts_list: PartsListConfig = Field(default_factory=PartsListConfig)
    """Configuration for parts list classification."""

    # Rotation symbol classifier settings
    rotation_symbol: RotationSymbolConfig = Field(default_factory=RotationSymbolConfig)
    """Configuration for rotation symbol classification."""

    # Arrow classifier settings
    arrow: ArrowConfig = Field(default_factory=ArrowConfig)
    """Configuration for arrow (arrowhead) classification."""

    # Bag number classifier settings
    bag_number: BagNumberConfig = Field(default_factory=BagNumberConfig)
    """Configuration for bag number classification."""

    # Background classifier settings
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    """Configuration for background classification."""

    # Divider classifier settings
    divider: DividerConfig = Field(default_factory=DividerConfig)
    """Configuration for divider classification."""

    # NewBag classifier settings (for numberless bag detection)
    new_bag: NewBagConfig = Field(default_factory=NewBagConfig)
    """Configuration for new bag classification."""

    # Step count classifier settings
    step_count: StepCountConfig = Field(default_factory=StepCountConfig)
    """Configuration for step count classification."""

    # SubAssembly classifier settings
    subassembly: SubAssemblyConfig = Field(default_factory=SubAssemblyConfig)
    """Configuration for subassembly classification."""

    # Trivia text classifier settings
    trivia_text: TriviaTextConfig = Field(default_factory=TriviaTextConfig)
    """Configuration for trivia/flavor text classification."""

    font_size_hints: FontSizeHints = Field(default_factory=FontSizeHints.empty)
    """Font size hints derived from analyzing all pages"""

    page_hints: PageHintCollection = Field(default_factory=PageHintCollection.empty)
    """Page type hints derived from analyzing all pages"""
