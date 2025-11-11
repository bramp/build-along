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

from .classification_result import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from .classifier import Classifier, classify_elements, classify_pages
from .label_classifier import LabelClassifier
from .page_number_classifier import PageNumberClassifier
from .part_count_classifier import PartCountClassifier
from .part_number_classifier import PartNumberClassifier
from .parts_image_classifier import PartsImageClassifier
from .parts_list_classifier import PartsListClassifier
from .progress_bar_classifier import ProgressBarClassifier
from .step_classifier import StepClassifier
from .step_number_classifier import StepNumberClassifier

__all__ = [
    "classify_elements",
    "classify_pages",
    "Candidate",
    "Classifier",
    "ClassificationResult",
    "ClassifierConfig",
    "LabelClassifier",
    "StepClassifier",
    "PageNumberClassifier",
    "PartCountClassifier",
    "PartNumberClassifier",
    "StepNumberClassifier",
    "PartsListClassifier",
    "PartsImageClassifier",
    "ProgressBarClassifier",
]

import logging
import os

_level = os.getenv("LOG_LEVEL")
if _level:
    try:
        # Only configure if no handlers are present so apps/tests can override.
        if not logging.getLogger().handlers:
            logging.basicConfig(level=getattr(logging, _level.upper(), logging.INFO))
    except Exception:
        # Fail-safe: never raise from import due to logging setup.
        pass
