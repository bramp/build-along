"""Common utilities for PDF extraction."""

from typing import Any


def round_floats(obj: Any, decimals: int = 2) -> Any:
    """Recursively round all floats in a nested structure to specified decimals.

    Args:
        obj: Any Python object (dict, list, tuple, float, or other)
        decimals: Number of decimal places to round to (default: 2)

    Returns:
        The same structure with all floats rounded
    """
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(round_floats(item, decimals) for item in obj)
    return obj
