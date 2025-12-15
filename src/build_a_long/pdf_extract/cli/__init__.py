"""CLI support for PDF extraction tool."""

from build_a_long.pdf_extract.validation import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    print_validation,
    validate_results,
)

from .config import ProcessingConfig, parse_arguments
from .io import (
    load_json,
    open_compressed,
    render_annotated_images,
    save_debug_json,
    save_manual_json,
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
from .unassigned_diagnostics import (
    UnassignedCategory,
    analyze_unassigned_blocks,
    print_unassigned_diagnostics,
)

__all__ = [
    "ProcessingConfig",
    "parse_arguments",
    "load_json",
    "open_compressed",
    "render_annotated_images",
    "save_debug_json",
    "save_manual_json",
    "save_raw_json",
    "DebugOutput",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
    "print_classification_debug",
    "print_font_hints",
    "print_histogram",
    "print_page_hierarchy",
    "print_summary",
    "print_validation",
    "validate_results",
    "UnassignedCategory",
    "analyze_unassigned_blocks",
    "print_unassigned_diagnostics",
]
