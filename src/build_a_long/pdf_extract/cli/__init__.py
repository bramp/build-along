"""CLI support for PDF extraction tool."""

from .config import ProcessingConfig, parse_arguments
from .io import render_annotated_images, save_classified_json, save_raw_json
from .reporting import (
    print_classification_debug,
    print_histogram,
    print_label_counts,
    print_page_hierarchy,
    print_summary,
)

__all__ = [
    "ProcessingConfig",
    "parse_arguments",
    "render_annotated_images",
    "save_classified_json",
    "save_raw_json",
    "print_classification_debug",
    "print_histogram",
    "print_label_counts",
    "print_page_hierarchy",
    "print_summary",
]
