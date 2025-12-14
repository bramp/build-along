"""Common utilities for PDF extraction."""

import json
from typing import Any


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


def transform_for_json(obj: Any, decimals: int = 2) -> Any:
    """Transform data for JSON serialization in a single pass.

    Combines rounding floats and reordering __tag__ to first position
    into one recursive traversal for better performance.

    Args:
        obj: Any Python object (dict, list, tuple, float, or other)
        decimals: Number of decimal places to round floats to (default: 2)

    Returns:
        The transformed structure ready for JSON serialization
    """
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        # Process all values recursively
        processed = {k: transform_for_json(v, decimals) for k, v in obj.items()}
        # If __tag__ exists, put it first
        if "__tag__" in processed:
            tag_value = processed.pop("__tag__")
            return {"__tag__": tag_value, **processed}
        return processed
    elif isinstance(obj, list):
        return [transform_for_json(item, decimals) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(transform_for_json(item, decimals) for item in obj)
    return obj


class SerializationMixin:
    """Mixin providing consistent to_dict() and to_json() serialization.

    Add this as a base class to any Pydantic BaseModel to get standardized
    serialization with:
    - by_alias=True (uses field aliases like __tag__)
    - exclude_none=True (omits None values)
    - Floats rounded to 2 decimal places
    - Tab indentation for JSON (configurable)

    Example:
        class MyModel(SerializationMixin, BaseModel):
            name: str
            value: float

        model = MyModel(name="test", value=3.14159)
        model.to_json()  # '{"name": "test", "value": 3.14}'
    """

    def to_dict(self, **kwargs: Any) -> dict:
        """Serialize to dict with proper defaults.

        Uses by_alias=True, exclude_none=True, rounds floats to 2 decimals,
        and ensures __tag__ appears first in each dict for readability.
        Override by passing explicit kwargs if different behavior is needed.
        """
        defaults: dict[str, Any] = {"by_alias": True, "exclude_none": True}
        defaults.update(kwargs)
        data = self.model_dump(**defaults)  # type: ignore[attr-defined]
        return transform_for_json(data)

    def to_json(self, *, indent: str | int | None = "\t", **kwargs: Any) -> str:
        """Serialize to JSON with proper defaults.

        Uses by_alias=True, exclude_none=True, rounds floats to 2 decimals,
        and uses tab indentation by default.
        Override by passing explicit kwargs if different behavior is needed.

        Args:
            indent: Indentation for pretty-printing (default: tab).
                    Use None for compact output.
            **kwargs: Additional arguments passed to model_dump()
        """
        data = self.to_dict(**kwargs)
        return json.dumps(data, indent=indent)
