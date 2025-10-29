"""Classifier package entry point and logging configuration.

This module exposes top-level helper(s) and sets up optional logging
configuration based on environment variables:

- LOG_LEVEL: Global log level (e.g., DEBUG, INFO). If set and logging
        is not already configured by the application, configure a basic
        handler at this level.

- CLASSIFIER_DEBUG: Topic-specific debug selector. Supported values:
        "page_number", "part_count", "step_number", "parts_list", "part_image", or "all".
        Each classifier module checks this to emit richer, structured debug logs.

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

from .classifier import classify_elements, classify_pages

__all__ = ["classify_elements", "classify_pages"]

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
