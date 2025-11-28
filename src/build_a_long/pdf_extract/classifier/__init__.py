"""Classifier package entry point and logging configuration.

This module exposes top-level helper(s) and sets up optional logging
configuration based on environment variables:

- LOG_LEVEL: Global log level (e.g., DEBUG, INFO). If set and logging
        is not already configured by the application, configure a basic
        handler at this level.

Classifier pipeline order
-------------------------
The classifier pipeline runs in a fixed order, enforced at initialization:

1) PageNumber → "page_number"
2) PartCount  → "part_count"
3) StepNumber → "step_number" (uses page number size as context)
4) PartsList  → "parts_list" (requires step_number and part_count)
5) PartsImage → "part_image" (requires parts_list and part_count)

Changing the order such that dependencies are not met will raise a ValueError.
"""

import logging
import os

from build_a_long.pdf_extract.classifier.bags import (
    BagNumberClassifier,
    NewBagClassifier,
)
from build_a_long.pdf_extract.classifier.pages import (
    PageHint,
    PageHintCollection,
)
from build_a_long.pdf_extract.classifier.pages.page_classifier import PageClassifier
from build_a_long.pdf_extract.classifier.pages.page_number_classifier import (
    PageNumberClassifier,
)
from build_a_long.pdf_extract.classifier.pages.progress_bar_classifier import (
    ProgressBarClassifier,
)
from build_a_long.pdf_extract.classifier.parts import (
    PartCountClassifier,
    PartNumberClassifier,
    PartsClassifier,
    PartsImageClassifier,
    PartsListClassifier,
    PieceLengthClassifier,
)
from build_a_long.pdf_extract.classifier.steps import (
    DiagramClassifier,
    StepClassifier,
    StepNumberClassifier,
)

from ..extractor.lego_page_elements import Page
from .batch_classification_result import BatchClassificationResult
from .candidate import Candidate
from .classification_result import ClassificationResult
from .classifier import Classifier, classify_elements, classify_pages
from .classifier_config import ClassifierConfig
from .label_classifier import LabelClassifier
from .removal_reason import RemovalReason
from .score import Score, Weight
from .text import FontSizeHints, TextHistogram

PageType = Page.PageType

__all__ = [
    "classify_elements",
    "classify_pages",
    "BagNumberClassifier",
    "BatchClassificationResult",
    "Candidate",
    "Classifier",
    "ClassificationResult",
    "ClassifierConfig",
    "DiagramClassifier",
    "FontSizeHints",
    "LabelClassifier",
    "NewBagClassifier",
    "PageHint",
    "PageHintCollection",
    "PageType",
    "RemovalReason",
    "Score",
    "StepClassifier",
    "PageNumberClassifier",
    "PageClassifier",
    "PartCountClassifier",
    "PartNumberClassifier",
    "PartsClassifier",
    "PieceLengthClassifier",
    "StepNumberClassifier",
    "PartsListClassifier",
    "PartsImageClassifier",
    "ProgressBarClassifier",
    "TextHistogram",
    "Weight",
]

_level = os.getenv("LOG_LEVEL")
if _level:
    try:
        # Only configure if no handlers are present so apps/tests can override.
        if not logging.getLogger().handlers:
            logging.basicConfig(level=getattr(logging, _level.upper(), logging.INFO))
    except Exception:
        # Fail-safe: never raise from import due to logging setup.
        pass
