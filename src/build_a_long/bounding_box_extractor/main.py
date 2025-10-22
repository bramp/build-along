import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

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

# Require PyMuPDF at import time. This purposefully fails fast if missing.
import fitz  # type: ignore  # PyMuPDF


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

    for element in page_data["elements"]:
        # Elements are typed PageElements now
        bbox = element.bbox
        bbox_tuple = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        color = "red" if isinstance(element, StepNumber) else "blue"
        draw.rectangle(bbox_tuple, outline=color, width=2)

    output_path = output_dir / f"page_{page_num:03d}.png"
    img.save(output_path)
    print(f"  Saved image with bboxes to {output_path}")


# TODO pdf_path should be Path-like
def extract_bounding_boxes(pdf_path: str, output_dir: Path | None) -> Dict[str, Any]:
    """
    Extract bounding boxes for instruction numbers, parts lists, and build steps
    from a given PDF file using PyMuPDF when available.

    Returns the extracted data dict for convenience (also written to JSON file).
    """
    print(f"Processing PDF: {pdf_path}")
    extracted_data: Dict[str, Any] = {"pages": []}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    doc = None
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        for page_index in range(num_pages):
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
        description="Extract bounding boxes from a PDF file."
    )
    parser.add_argument("pdf_path", help="The path to the PDF file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save images with drawn bounding boxes.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return 2

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    extract_bounding_boxes(str(pdf_path), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
