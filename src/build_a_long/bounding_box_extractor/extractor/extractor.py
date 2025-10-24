from typing import Any, Dict, List

import pymupdf

from build_a_long.bounding_box_extractor.extractor.bbox import BBox
from build_a_long.bounding_box_extractor.extractor.hierarchy import (
    build_hierarchy_from_elements,
)
from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Drawing,
    PathElement,
    StepNumber,
    Text,
    Unknown,
)


def _classify_text(text: str) -> str:
    """Return a coarse label for a text block.

    - "instruction_number" if the text is just a number (e.g., step number)
    - "parts_list" if it looks like a parts header
    - "text" otherwise
    """
    t = text.strip()
    if t.isdigit():
        return "instruction_number"
    lower = t.lower()
    if "parts" in lower or "spare" in lower:
        return "parts_list"
    return "text"


def extract_bounding_boxes(
    pdf_path: str,
    start_page: int | None = None,
    end_page: int | None = None,
    include_types: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Extract bounding boxes for instruction numbers, parts lists, and build steps
    from a given PDF file using PyMuPDF when available.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save images and JSON files (required for output)
        start_page: First page to process (1-indexed), None for first page
        end_page: Last page to process (1-indexed, inclusive), None for last page

    Returns the extracted data dict for convenience (also written to JSON file).
    """
    print(f"Processing PDF: {pdf_path}")
    extracted_data: Dict[str, Any] = {"pages": []}

    # Initialize page range variables
    doc = None
    try:
        # TODO Do I need to call doc.close() later?
        doc = pymupdf.open(pdf_path)
        num_pages = len(doc)

        # TODO Let's remove the zero based page indexing variables. It just
        # .     confuses things.
        # Determine page range (convert to 0-indexed)
        first_page = (start_page - 1) if start_page is not None else 0
        last_page = (end_page - 1) if end_page is not None else (num_pages - 1)

        # Validate and clamp page range
        first_page = max(0, min(first_page, num_pages - 1))
        last_page = max(0, min(last_page, num_pages - 1))

        if first_page > last_page:
            print(f"Warning: Invalid page range {start_page}-{end_page}")
            return extracted_data

        print(f"Processing pages {first_page + 1}-{last_page + 1} of {num_pages}")

        for page_index in range(first_page, last_page + 1):
            page = doc[page_index]
            page_num = page_index + 1
            print(f"  Processing page {page_num}/{num_pages}")

            page_data: Dict[str, Any] = {"page_number": page_num, "elements": []}
            typed_elements = []  # In-memory typed elements for this page

            # Use rawdict to get both text and image blocks with bboxes
            raw = page.get_text("rawdict")
            blocks: List[Dict[str, Any]] = (
                raw.get("blocks", []) if isinstance(raw, dict) else []
            )

            if include_types is None or "text" in include_types:
                for bi, b in enumerate(blocks):
                    btype = b.get("type")  # 0=text, 1=image
                    if btype == 0:  # Text block
                        for li, line in enumerate(b.get("lines", [])):
                            for si, span in enumerate(line.get("spans", [])):
                                sbbox = span.get("bbox", [0, 0, 0, 0])
                                try:
                                    nbbox = BBox(
                                        x0=float(sbbox[0]),
                                        y0=float(sbbox[1]),
                                        x1=float(sbbox[2]),
                                        y1=float(sbbox[3]),
                                    )
                                except Exception:
                                    nbbox = BBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)

                                text = span.get("text", "")
                                if not isinstance(text, str):
                                    text = ""

                                print(
                                    f"    Found text span: '{text}' with bbox {nbbox}"
                                )

                                label = _classify_text(text)
                                if (
                                    label == "instruction_number"
                                    and text.strip().isdigit()
                                ):
                                    typed_elements.append(
                                        StepNumber(bbox=nbbox, value=int(text.strip()))
                                    )
                                else:
                                    # Create a Text element for regular text
                                    typed_elements.append(
                                        Text(
                                            bbox=nbbox,
                                            content=text,
                                            label=(
                                                "parts_list"
                                                if label == "parts_list"
                                                else None
                                            ),
                                        )
                                    )

            if include_types is None or "image" in include_types:
                for bi, b in enumerate(blocks):
                    btype = b.get("type")  # 0=text, 1=image
                    if btype == 1:
                        bbox = b.get("bbox", [0, 0, 0, 0])
                        try:
                            nbbox = BBox(
                                x0=float(bbox[0]),
                                y0=float(bbox[1]),
                                x1=float(bbox[2]),
                                y1=float(bbox[3]),
                            )
                            typed_elements.append(
                                Drawing(bbox=nbbox, image_id=f"image_{bi}")
                            )
                        except Exception:
                            pass  # Ignore invalid bbox

            if include_types is None or "drawing" in include_types:
                for bi, b in enumerate(blocks):
                    btype = b.get("type")  # 0=text, 1=image
                    if btype not in [0, 1]:
                        bbox = b.get("bbox", [0, 0, 0, 0])
                        try:
                            nbbox = BBox(
                                x0=float(bbox[0]),
                                y0=float(bbox[1]),
                                x1=float(bbox[2]),
                                y1=float(bbox[3]),
                            )
                            typed_elements.append(
                                Unknown(
                                    bbox=nbbox,
                                    raw_type=f"unknown_{btype}",
                                    source_id=f"unknown_{bi}",
                                    btype=btype,
                                )
                            )
                        except Exception:
                            pass  # Ignore invalid bbox

            # Now get drawings (paths)
            if include_types is None or "path" in include_types:
                drawings = page.get_drawings()
                for d in drawings:
                    drect = d["rect"]
                    try:
                        nbbox = BBox(
                            x0=float(drect.x0),
                            y0=float(drect.y0),
                            x1=float(drect.x1),
                            y1=float(drect.y1),
                        )
                        typed_elements.append(PathElement(bbox=nbbox))
                    except Exception:
                        pass  # Ignore invalid bbox

            # Build containment hierarchy from typed elements and store typed structures
            roots = build_hierarchy_from_elements(typed_elements)
            page_data["elements"] = typed_elements  # typed Element list
            page_data["hierarchy"] = roots  # tuple[ElementNode, ...]

            # Placeholder for build step identification (future work)
            extracted_data["pages"].append(page_data)

            # Note: drawing PNGs is handled by the caller now.

    except Exception as e:
        print(f"An error occurred while processing '{pdf_path}': {e}")
    finally:
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass

    # Note: JSON persistence is handled by the caller now. We just return data.
    return extracted_data
