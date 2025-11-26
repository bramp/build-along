"""CLI support for PDF extraction tool."""

from .config import ProcessingConfig, parse_arguments
from .io import (
    load_json,
    open_compressed,
    render_annotated_images,
    save_debug_json,
    save_pages_json,
    save_raw_json,
)
from .output_models import DebugOutput
from .reporting import (
    print_classification_debug,
    print_font_hints,
    print_histogram,
    print_page_hierarchy,
    print_summary,
)

__all__ = [
    "ProcessingConfig",
    "parse_arguments",
    "load_json",
    "open_compressed",
    "render_annotated_images",
    "save_debug_json",
    "save_pages_json",
    "save_raw_json",
    "DebugOutput",
    "print_classification_debug",
    "print_font_hints",
    "print_histogram",
    "print_page_hierarchy",
    "print_summary",
]
