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


def remove_empty_lists(obj: Any) -> Any:
    """Recursively remove empty lists from a nested structure.

    Args:
        obj: Any Python object (dict, list, or other)

    Returns:
        The same structure with all empty lists removed from dicts
    """
    if isinstance(obj, dict):
        return {
            k: remove_empty_lists(v)
            for k, v in obj.items()
            if not (isinstance(v, list) and len(v) == 0)
        }
    elif isinstance(obj, list):
        return [remove_empty_lists(item) for item in obj]
    return obj
