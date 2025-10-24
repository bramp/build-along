import logging
from typing import Any, Dict, Set

import pymupdf

from build_a_long.bounding_box_extractor.extractor.bbox import BBox

from build_a_long.bounding_box_extractor.extractor.hierarchy import (
    build_hierarchy_from_elements,
)
from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Drawing,
    Image,
    Root,
    Text,
)
from build_a_long.bounding_box_extractor.extractor.pymupdf_types import (
    BBoxTuple,
    RawDict,
)

logger = logging.getLogger("extractor")


def extract_bounding_boxes(
    pdf_path: str,
    start_page: int | None = None,
    end_page: int | None = None,
    include_types: Set[str] = {"text", "image", "drawing"},
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
    logger.info(f"Processing PDF: {pdf_path}")
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
            logger.warning("Invalid page range %s-%s", str(start_page), str(end_page))
            return extracted_data

        logger.info(
            "Processing pages %s-%s of %s", first_page + 1, last_page + 1, num_pages
        )

        for page_index in range(first_page, last_page + 1):
            page = doc[page_index]
            page_num = page_index + 1
            logger.info("Processing page %s/%s", page_num, num_pages)

            page_data: Dict[str, Any] = {"page_number": page_num, "elements": []}
            typed_elements = []  # In-memory typed elements for this page

            # Use rawdict to get both text and image blocks with bboxes
            raw: RawDict = page.get_text("rawdict")  # type: ignore[assignment]
            assert isinstance(raw, dict)

            # See https://pymupdf.readthedocs.io/en/latest/textpage.html#page-dictionary
            for b in raw.get("blocks", []):
                assert isinstance(b, dict)

                # See https://pymupdf.readthedocs.io/en/latest/textpage.html#block-dictionaries
                bi: int | None = b.get("number")
                btype: int | None = b.get("type")  # 0=text, 1=image

                if btype == 0:  # Text block
                    if "text" not in include_types:
                        continue

                    for line in b.get("lines", []):
                        for si, span in enumerate(line.get("spans", [])):
                            sbbox: BBoxTuple = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                            nbbox = BBox(
                                x0=float(sbbox[0]),
                                y0=float(sbbox[1]),
                                x1=float(sbbox[2]),
                                y1=float(sbbox[3]),
                            )

                            text: str = span.get("text", "")
                            logger.debug(
                                "Found text %s %r with bbox %s", bi, text, nbbox
                            )

                            # Create a Text element for regular text
                            typed_elements.append(
                                Text(
                                    bbox=nbbox,
                                    content=text,
                                    label=None,
                                )
                            )

                elif btype == 1:
                    if "image" not in include_types:
                        continue

                    bbox: BBoxTuple = b.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    nbbox = BBox(
                        x0=float(bbox[0]),
                        y0=float(bbox[1]),
                        x1=float(bbox[2]),
                        y1=float(bbox[3]),
                    )
                    typed_elements.append(Image(bbox=nbbox, image_id=f"image_{bi}"))

                    logger.debug("Found image %s with %s", bi, nbbox)

                else:
                    logger.warning(
                        "Skipping block with unsupported type %s at index %s", btype, bi
                    )

            # Now get drawings (paths)
            if "drawing" in include_types:
                drawings = page.get_drawings()
                for d in drawings:
                    drect = d["rect"]
                    nbbox = BBox(
                        x0=float(drect.x0),
                        y0=float(drect.y0),
                        x1=float(drect.x1),
                        y1=float(drect.y1),
                    )
                    typed_elements.append(Drawing(bbox=nbbox))
                    logger.debug("Found drawing with %s", nbbox)

            # Create a Root element encompassing the entire page
            page_rect = page.rect
            root_bbox = BBox(
                x0=float(page_rect.x0),
                y0=float(page_rect.y0),
                x1=float(page_rect.x1),
                y1=float(page_rect.y1),
            )
            root_element = Root(bbox=root_bbox)

            # Build containment hierarchy from typed elements
            roots = build_hierarchy_from_elements(typed_elements)
            page_data["root"] = root_element
            page_data["elements"] = typed_elements  # typed Element list
            page_data["hierarchy"] = roots  # tuple[Element, ...]

            # Placeholder for build step identification (future work)
            extracted_data["pages"].append(page_data)

    except Exception:
        # logger.exception will include the traceback automatically
        logger.exception("An error occurred while processing %s", pdf_path)

    finally:
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass

    return extracted_data
