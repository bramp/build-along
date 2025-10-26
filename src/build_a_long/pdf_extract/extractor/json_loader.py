"""Utilities to load PageData and elements from a simple JSON schema.

Schema (version 1):

{
    "page_number": 1,                 # int
    "bbox": [x0, y0, x1, y1],         # page bounds (also accepts {"x0":..., "y0":..., "x1":..., "y1":...})
    "elements": [
        {"_type_": "text",    "bbox": [..], "text": "9",  "font_name": "...", "font_size": 12.0},
        {"_type_": "image",   "bbox": [..], "image_id": "image_1"},
        {"_type_": "drawing", "bbox": [..]}
    ]
}

Only the shown fields are supported. Optional fields may be omitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping

import json

from build_a_long.pdf_extract.extractor.bbox import _bbox_decoder
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    Image,
    Text,
    PageElement,
)
from build_a_long.pdf_extract.extractor.extractor import PageData


def _element_from_json(d: Mapping[str, Any]) -> PageElement:
    """Parse a single element from JSON using type-based dispatch.

    Discriminator key is "_type_" (preferred).
    """
    etype = d.get("_type_")
    if etype not in {"text", "image", "drawing"}:
        raise ValueError(f"Unsupported element type: {etype}")

    dm = dict(d)
    dm.pop("_type_", None)

    if etype == "text":
        return Text.from_dict(dm)  # type: ignore[attr-defined]
    if etype == "image":
        return Image.from_dict(dm)  # type: ignore[attr-defined]
    return Drawing.from_dict(dm)  # type: ignore[attr-defined]


def load_page_from_dict(data: Mapping[str, Any]) -> PageData:
    """Create a PageData from a parsed JSON mapping using the v1 schema.

    Args:
        data: Parsed JSON object (dict-like)

    Returns:
        PageData instance
    """
    try:
        page_number = int(data["page_number"])  # type: ignore[index]
    except Exception as e:  # pragma: no cover - defensive
        raise ValueError("'page_number' must be provided as an int") from e

    bbox_val = data.get("bbox", [0.0, 0.0, 0.0, 0.0])
    bbox = _bbox_decoder(bbox_val)

    raw_elements = data.get("elements", [])
    if not isinstance(raw_elements, list):
        raise ValueError("'elements' must be a list")

    elements: List[PageElement] = [_element_from_json(e) for e in raw_elements]

    return PageData(page_number=page_number, elements=elements, bbox=bbox)


def load_page_from_json(path: str | Path) -> PageData:
    """Load PageData from a JSON file on disk.

    Args:
        path: Path to a JSON file following the schema described above.

    Returns:
        PageData instance
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")
    return load_page_from_dict(data)


__all__ = ["load_page_from_json", "load_page_from_dict"]
