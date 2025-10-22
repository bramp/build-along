import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from build_a_long.bounding_box_extractor.bbox import BBox

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


# TODO pdf_path should be Path-like
def extract_bounding_boxes(pdf_path: str) -> Dict[str, Any]:
    """
    Extract bounding boxes for instruction numbers, parts lists, and build steps
    from a given PDF file using PyMuPDF when available.

    Returns the extracted data dict for convenience (also written to JSON file).
    """
    print(f"Processing PDF: {pdf_path}")
    extracted_data: Dict[str, Any] = {"pages": []}

    doc = None
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        for page_index in range(num_pages):
            page = doc[page_index]
            page_num = page_index + 1
            print(f"  Processing page {page_num}/{num_pages}")

            page_data: Dict[str, Any] = {"page_number": page_num, "elements": []}

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
                    page_data["elements"].append(
                        {
                            "type": label,
                            "bbox": [nbbox.x0, nbbox.y0, nbbox.x1, nbbox.y1],
                            "content": text,
                            "id": f"text_{bi}",
                        }
                    )
                elif btype == 1:
                    # Image block
                    page_data["elements"].append(
                        {
                            "type": "image",
                            "bbox": [nbbox.x0, nbbox.y0, nbbox.x1, nbbox.y1],
                            "id": f"image_{bi}",
                        }
                    )
                else:
                    # Other block types (e.g., drawings) - skip for now, could be used for build steps later.
                    pass

            # Placeholder for build step identification (future work)
            extracted_data["pages"].append(page_data)

    except Exception as e:
        print(f"An error occurred while processing '{pdf_path}': {e}")
    finally:
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass

    # Write JSON output
    output_filename = pdf_path.replace(".pdf", ".json")
    with open(output_filename, "w") as f:
        json.dump(extracted_data, f, indent=4)
    print(f"Extracted data saved to {output_filename}")
    return extracted_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract bounding boxes from a PDF file."
    )
    parser.add_argument("pdf_path", help="The path to the PDF file.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return 2

    extract_bounding_boxes(str(pdf_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
