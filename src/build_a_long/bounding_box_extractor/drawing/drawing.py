from pathlib import Path
from typing import Tuple

import fitz  # type: ignore  # PyMuPDF
from PIL import Image, ImageDraw  # type: ignore

from build_a_long.bounding_box_extractor.extractor.hierarchy import ElementNode
from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Drawing,
    Text,
    Unknown,
)


def draw_and_save_bboxes(
    page: fitz.Page,
    hierarchy: Tuple[ElementNode, ...],
    output_dir: Path,
    page_num: int,
    image_dpi: int = 150,
):
    """
    Draws bounding boxes from a hierarchy on the PDF page image and saves it.
    Colors are based on nesting depth, and element types are labeled.
    """
    # Render page to an image
    pix = page.get_pixmap(colorspace=fitz.csRGB, dpi=image_dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)

    # Get page dimensions for scaling
    page_rect = page.rect
    scale_x = pix.width / page_rect.width
    scale_y = pix.height / page_rect.height

    # Colors for different nesting depths (cycles through this list)
    depth_colors = ["red", "green", "blue", "yellow", "purple", "orange"]

    def _draw_node(node: ElementNode, depth: int) -> None:
        """Recursively draw a node and its children."""
        element = node.element
        bbox = element.bbox

        # Scale the bounding box
        scaled_bbox = (
            bbox.x0 * scale_x,
            bbox.y0 * scale_y,
            bbox.x1 * scale_x,
            bbox.y1 * scale_y,
        )

        # Determine color based on depth
        color = depth_colors[depth % len(depth_colors)]

        # Draw the bounding box
        draw.rectangle(scaled_bbox, outline=color, width=2)

        # Draw the element type text
        label = element.__class__.__name__
        if isinstance(element, Drawing):
            if element.image_id:
                label = f"{label} ({element.image_id})"
        elif isinstance(element, Text):
            # For Text elements, show the actual text content
            content = element.content.strip()
            if len(content) > 50:  # Truncate long text
                content = content[:47] + "..."
            label = f"{label}: {content}"
        elif isinstance(element, Unknown):
            if element.btype is not None:
                btype_map = {0: "text", 1: "image", 2: "drawing"}
                btype_str = btype_map.get(element.btype, f"btype={element.btype}")
                label = f"{label} ({btype_str})"
            if element.source_id:
                # source_id is like "text_bi_li_si" or "unknown_bi"
                parts = element.source_id.split("_")
                if len(parts) > 1:
                    label = f"{label} bi={parts[1]}"

        # Below bottom-left
        text_position = (scaled_bbox[0], scaled_bbox[3] + 2)
        draw.text(text_position, label, fill=color)

        # Recursively draw children
        for child in node.children:
            _draw_node(child, depth + 1)

    # Start traversal from root nodes
    for root_node in hierarchy:
        _draw_node(root_node, 0)

    output_path = output_dir / f"page_{page_num:03d}.png"
    img.save(output_path)
    print(f"  Saved image with bboxes to {output_path}")
