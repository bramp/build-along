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
from .classification_result import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from .classifier import Classifier, classify_elements, classify_pages
from .label_classifier import LabelClassifier
from .page_hints import PageHint, PageHints
from .page_number_classifier import PageNumberClassifier
from .progress_bar_classifier import ProgressBarClassifier

PageType = Page.PageType

__all__ = [
    "classify_elements",
    "classify_pages",
    "BagNumberClassifier",
    "Candidate",
    "Classifier",
    "ClassificationResult",
    "ClassifierConfig",
    "DiagramClassifier",
    "LabelClassifier",
    "NewBagClassifier",
    "PageHint",
    "PageHints",
    "PageType",
    "StepClassifier",
    "PageNumberClassifier",
    "PartCountClassifier",
    "PartNumberClassifier",
    "PieceLengthClassifier",
    "StepNumberClassifier",
    "PartsListClassifier",
    "PartsImageClassifier",
    "ProgressBarClassifier",
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
