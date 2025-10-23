import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw  # type: ignore

from build_a_long.bounding_box_extractor.bbox import BBox
from build_a_long.bounding_box_extractor.hierarchy import (
    build_hierarchy_from_elements,
)
from build_a_long.bounding_box_extractor.page_elements import (
    StepNumber,
    Drawing,
    Unknown,
)

import fitz  # type: ignore  # PyMuPDF


def parse_page_range(page_str: str) -> Tuple[int | None, int | None]:
    """Parse a page range string into start and end page numbers.

    Supported formats:
    - "5": Single page (returns 5, 5)
    - "5-10": Page range from 5 to 10 (returns 5, 10)
    - "10-": From page 10 to end (returns 10, None)
    - "-5": From start to page 5 (returns None, 5)

    Args:
        page_str: The page range string to parse.

    Returns:
        A tuple of (start_page, end_page), where None indicates an unbounded range.

    Raises:
        ValueError: If the page range format is invalid or contains invalid numbers.
    """
    # TODO in future support accepting lists, e.g "1, 2, 3" or "1-3,5,7-9"

    page_str = page_str.strip()
    if not page_str:
        raise ValueError("Page range cannot be empty")

    # Check for range syntax
    if "-" in page_str:
        parts = page_str.split("-", 1)
        start_str = parts[0].strip()
        end_str = parts[1].strip()

        # Handle "-5" format (start to page 5)
        if not start_str:
            if not end_str:
                raise ValueError(
                    "Invalid page range: '-'. At least one page number required."
                )
            try:
                end_page = int(end_str)
                # Reject negative numbers - they look like "-5" but are actually negative
                if end_str.startswith("-"):
                    raise ValueError(f"Page number must be >= 1, got {end_page}")
                if end_page < 1:
                    raise ValueError(f"Page number must be >= 1, got {end_page}")
                return None, end_page
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid end page number: '{end_str}'")
                raise

        # Handle "10-" format (page 10 to end)
        if not end_str:
            try:
                start_page = int(start_str)
                if start_page < 1:
                    raise ValueError(f"Page number must be >= 1, got {start_page}")
                return start_page, None
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid start page number: '{start_str}'")
                raise

        # Handle "5-10" format (explicit range)
        try:
            start_page = int(start_str)
        except ValueError:
            raise ValueError(f"Invalid start page number: '{start_str}'")

        try:
            end_page = int(end_str)
        except ValueError:
            raise ValueError(f"Invalid end page number: '{end_str}'")

        if start_page < 1:
            raise ValueError(f"Start page must be >= 1, got {start_page}")
        if end_page < 1:
            raise ValueError(f"End page must be >= 1, got {end_page}")
        if start_page > end_page:
            raise ValueError(
                f"Start page ({start_page}) cannot be greater than end page ({end_page})"
            )
        return start_page, end_page
    else:
        # Single page number
        try:
            page_num = int(page_str)
            if page_num < 1:
                raise ValueError(f"Page number must be >= 1, got {page_num}")
            return page_num, page_num
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid page number: '{page_str}'")
            raise


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


def draw_and_save_bboxes(
    page: fitz.Page,
    page_data: Dict[str, Any],
    output_dir: Path,
    page_num: int,
    image_dpi: int = 150,
):
    """
    Draws bounding boxes on the PDF page image and saves the image to disk.
    """
    # Render page to an image
    pix = page.get_pixmap(colorspace=fitz.csRGB, dpi=image_dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)

    # Get page dimensions for scaling
    page_rect = page.rect
    scale_x = pix.width / page_rect.width
    scale_y = pix.height / page_rect.height

    for element in page_data["elements"]:
        # Elements are typed PageElements now
        bbox = element.bbox
        scaled_bbox = (
            bbox.x0 * scale_x,
            bbox.y0 * scale_y,
            bbox.x1 * scale_x,
            bbox.y1 * scale_y,
        )
        color = "red" if isinstance(element, StepNumber) else "blue"
        draw.rectangle(scaled_bbox, outline=color, width=2)

    output_path = output_dir / f"page_{page_num:03d}.png"
    img.save(output_path)
    print(f"  Saved image with bboxes to {output_path}")


# TODO pdf_path should be Path-like
def extract_bounding_boxes(
    pdf_path: str,
    output_dir: Path | None,
    start_page: int | None = None,
    end_page: int | None = None,
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

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize page range variables
    first_page = 0
    last_page = 0
    doc = None
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)

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

            for bi, b in enumerate(blocks):
                btype = b.get("type")  # 0=text, 1=image
                bbox = b.get("bbox", [0, 0, 0, 0])
                # Normalize bbox to list of floats
                try:
                    nbbox = BBox(
                        x0=float(bbox[0]),
                        y0=float(bbox[1]),
                        x1=float(bbox[2]),
                        y1=float(bbox[3]),
                    )
                except Exception:
                    nbbox = BBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)

                if btype == 0:
                    # Concatenate spans' text
                    text_fragments: List[str] = []
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            t = span.get("text", "")
                            if isinstance(t, str):
                                text_fragments.append(t)
                    text = "".join(text_fragments)
                    label = _classify_text(text)
                    # Build typed element directly
                    if label == "instruction_number" and text.strip().isdigit():
                        typed_elements.append(
                            StepNumber(bbox=nbbox, value=int(text.strip()))
                        )
                    else:
                        typed_elements.append(
                            Unknown(
                                bbox=nbbox,
                                label="parts_list" if label == "parts_list" else None,
                                raw_type=label,
                                content=text,
                                source_id=f"text_{bi}",
                            )
                        )
                elif btype == 1:
                    # Image block -> build typed element directly
                    typed_elements.append(Drawing(bbox=nbbox, image_id=f"image_{bi}"))
                else:
                    # Other block types (e.g., drawings) - skip for now, could be used for build steps later.
                    pass

            # Build containment hierarchy from typed elements and store typed structures
            roots = build_hierarchy_from_elements(typed_elements)
            page_data["elements"] = typed_elements  # typed Element list
            page_data["hierarchy"] = roots  # tuple[ElementNode, ...]

            # Placeholder for build step identification (future work)
            extracted_data["pages"].append(page_data)

            # Draw and save bounding boxes if output_dir is provided
            if output_dir:
                draw_and_save_bboxes(page, page_data, output_dir, page_num)

    except Exception as e:
        print(f"An error occurred while processing '{pdf_path}': {e}")
    finally:
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass

    # JSON serialization: auto-serialize dataclasses using dataclasses.asdict()
    def _element_to_json(ele: Any) -> Dict[str, Any]:
        """Convert a PageElement to a JSON-friendly dict using asdict()."""
        data = asdict(ele)
        # Add a type discriminator for deserialization if needed later
        data["__type__"] = ele.__class__.__name__
        return data

    def _node_to_json(node: Any) -> Dict[str, Any]:
        """Convert an ElementNode to JSON recursively."""
        return {
            "element": _element_to_json(node.element),
            "children": [_node_to_json(c) for c in node.children],
        }

    json_data: Dict[str, Any] = {"pages": []}
    for page in extracted_data["pages"]:
        json_page: Dict[str, Any] = {"page_number": page["page_number"]}
        json_page["elements"] = [_element_to_json(e) for e in page["elements"]]
        if "hierarchy" in page:
            json_page["hierarchy"] = [_node_to_json(n) for n in page["hierarchy"]]
        json_data["pages"].append(json_page)

    output_json_path = (
        Path(pdf_path).with_suffix(".json")
        if output_dir is None
        else output_dir / (Path(pdf_path).stem + ".json")
    )
    try:
        with open(output_json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print(f"Extracted data saved to {output_json_path}")
    except Exception as e:
        print(f"Warning: failed to write JSON output to {output_json_path}: {e}")
    return extracted_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract bounding boxes from a PDF file and export images/JSON for debugging."
    )
    parser.add_argument("pdf_path", help="The path to the PDF file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save images and JSON files. Defaults to same directory as PDF.",
    )
    parser.add_argument(
        "--pages",
        type=str,
        help='Page range to process (1-indexed), e.g., "5", "5-10", "10-" (from 10 to end), or "-5" (from 1 to 5). Defaults to all pages.',
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return 2

    # Default output directory to same directory as PDF
    output_dir = args.output_dir if args.output_dir else pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse page range
    start_page = None
    end_page = None
    if args.pages:
        try:
            start_page, end_page = parse_page_range(args.pages)
        except ValueError as e:
            print(f"Error: {e}")
            return 2

    extract_bounding_boxes(str(pdf_path), output_dir, start_page, end_page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
