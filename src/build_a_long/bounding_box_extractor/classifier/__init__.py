"""Classifier package entry point and logging configuration.

This module exposes top-level helper(s) and sets up optional logging
configuration based on environment variables:

- LOG_LEVEL: Global log level (e.g., DEBUG, INFO). If set and logging
        is not already configured by the application, configure a basic
        handler at this level.

- CLASSIFIER_DEBUG: Topic-specific debug selector. Supported values:
        "page_number", "part_count", "step_number", "parts_list", or "all".
        Each classifier module checks this to emit richer, structured debug logs.
"""

from .classifier import classify_elements

__all__ = ["classify_elements"]

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
