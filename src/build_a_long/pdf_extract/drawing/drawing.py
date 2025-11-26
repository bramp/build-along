import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.drawing.path_renderer import (
    draw_dashed_rectangle,
    draw_path_items,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.hierarchy import build_hierarchy_from_blocks
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing as DrawingBlock,
)

logger = logging.getLogger(__name__)


@dataclass
class DrawableItem:
    """A unified structure for things to draw on the page."""

    bbox: BBox
    """The bounding box to draw."""

    label: str
    """Label text to display."""

    is_element: bool
    """True if this represents a LegoPageElement."""

    is_winner: bool
    """True if this is a winning element/block."""

    is_removed: bool
    """True if this block was removed."""

    depth: int = 0
    """Nesting depth for color selection (computed later)."""


def _create_drawable_items(
    result: ClassificationResult,
    *,
    draw_blocks: bool,
    draw_elements: bool,
    draw_deleted: bool,
    debug_candidates_label: str | None = None,
) -> list[DrawableItem]:
    """Create a unified list of items to draw.

    Args:
        result: Classification result containing blocks and elements
        draw_blocks: If True, include PDF blocks
        draw_elements: If True, include LEGO page elements
        draw_deleted: If True, include removed/non-winner items
        debug_candidates_label: If provided, only include candidates with this label

    Returns:
        List of DrawableItem objects ready to be rendered
    """
    items: list[DrawableItem] = []
    element_source_block_ids: set[int] = set()

    # Build set of chosen elements (elements in the final Page hierarchy)
    chosen_elements: set[int] = set()
    if result.page:
        for element in result.page.iter_elements():
            chosen_elements.add(id(element))

    # Add elements first and track their source block IDs
    if draw_elements:
        all_candidates = result.get_all_candidates()
        # Filter to specific label if requested
        if debug_candidates_label:
            all_candidates = {
                debug_candidates_label: all_candidates.get(debug_candidates_label, [])
            }

        for _, candidates in all_candidates.items():
            for candidate in candidates:
                is_constructed = candidate.constructed is not None
                # Check if this constructed element is in the final Page hierarchy
                is_winner = (
                    is_constructed and id(candidate.constructed) in chosen_elements
                )

                if not is_winner and not draw_deleted:
                    continue

                label_suffix = "" if is_winner else " [NOT WINNER]"
                label = f"[{candidate.label}]{label_suffix}"

                # Element without source block (e.g., Step, Page)
                items.append(
                    DrawableItem(
                        bbox=candidate.bbox,
                        label=label,
                        is_element=True,
                        is_winner=is_winner,
                        is_removed=False,
                    )
                )

                if candidate.source_blocks:
                    # Element with source blocks - track them and add as element
                    for source_block in candidate.source_blocks:
                        element_source_block_ids.add(source_block.id)

    # Add regular blocks (skip those that will be drawn as elements)
    if draw_blocks:
        for block in result.blocks:
            # Skip blocks that are source blocks for elements we're drawing
            if block.id in element_source_block_ids:
                continue

            is_removed = result.is_removed(block)
            if is_removed and not draw_deleted:
                continue

            # Get label from successfully constructed candidate for this block
            block_label = result.get_label(block)
            label_suffix = " [REMOVED]" if is_removed else ""
            label_text = block_label or str(block)

            label = f"ID: {block.id} {label_text}{label_suffix}"

            items.append(
                DrawableItem(
                    bbox=block.bbox,
                    label=label,
                    is_element=False,
                    is_winner=not is_removed,
                    is_removed=is_removed,
                )
            )

    return items


def _draw_item(
    draw: ImageDraw.ImageDraw,
    item: DrawableItem,
    depth_colors: list[str],
    scale_x: float,
    scale_y: float,
) -> None:
    """Draw a single item on the image.

    Args:
        draw: PIL ImageDraw object
        item: The item to draw
        depth_colors: List of colors to cycle through
        scale_x: X scaling factor
        scale_y: Y scaling factor
    """
    bbox = item.bbox
    scaled_bbox = (
        bbox.x0 * scale_x,
        bbox.y0 * scale_y,
        bbox.x1 * scale_x,
        bbox.y1 * scale_y,
    )

    color = depth_colors[item.depth % len(depth_colors)]

    # Determine drawing style
    if item.is_element:
        # Elements get thicker lines
        if item.is_winner:
            draw.rectangle(scaled_bbox, outline=color, width=2)
        else:
            # Non-winners get dashed thick lines
            draw_dashed_rectangle(draw, scaled_bbox, outline=color, width=2)
    else:
        # Regular blocks
        if item.is_removed:
            draw_dashed_rectangle(draw, scaled_bbox, outline=color, width=2)
        else:
            draw.rectangle(scaled_bbox, outline=color, width=1)

    # Draw label
    if item.depth % 2 == 0:
        text_position = (scaled_bbox[0], scaled_bbox[3] + 2)
        draw.text(text_position, item.label, fill=color)
    else:
        text_bbox = draw.textbbox((0, 0), item.label)
        text_width = text_bbox[2] - text_bbox[0]
        text_position = (scaled_bbox[2] - text_width, scaled_bbox[3] + 2)
        draw.text(text_position, item.label, fill=color)


def draw_and_save_bboxes(
    page: pymupdf.Page,
    result: ClassificationResult,
    output_path: Path,
    *,
    draw_blocks: bool = False,
    draw_elements: bool = False,
    draw_deleted: bool = False,
    draw_drawings: bool = False,
    debug_candidates_label: str | None = None,
) -> None:
    """
    Draws bounding boxes from blocks on the PDF page image and saves it.
    Colors are based on nesting depth (calculated via bbox containment).

    Args:
        page: PyMuPDF page to render
        result: ClassificationResult containing labels and blocks
        output_path: Where to save the output image
        draw_blocks: If True, render classified PDF blocks.
        draw_elements: If True, render classified LEGO page elements.
        draw_deleted: If True, also render blocks marked as deleted.
        draw_drawings: If True, render the actual drawing paths.
        debug_candidates_label: If provided, only render candidates with this label.
    """
    image_dpi = 150

    # Render page to an image
    pix = page.get_pixmap(colorspace=pymupdf.csRGB, dpi=image_dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)

    # Get page dimensions for scaling
    page_rect = page.rect
    scale_x = pix.width / page_rect.width
    scale_y = pix.height / page_rect.height

    # Colors for different nesting depths (cycles through this list)
    depth_colors = ["red", "green", "blue", "yellow", "purple", "orange"]

    # Create unified list of items to draw
    items = _create_drawable_items(
        result,
        draw_blocks=draw_blocks,
        draw_elements=draw_elements,
        draw_deleted=draw_deleted,
        debug_candidates_label=debug_candidates_label,
    )

    # Build hierarchy for depth calculation directly from DrawableItems
    hierarchy = build_hierarchy_from_blocks(items)

    # Compute and store depth in each item
    for item in items:
        item.depth = hierarchy.get_depth(item)

    # Draw all items
    for item in items:
        _draw_item(draw, item, depth_colors, scale_x, scale_y)

    # Draw actual drawing paths if requested
    if draw_drawings:
        drawings_rendered = 0
        clipped_count = 0
        for block in result.page_data.blocks:
            if isinstance(block, DrawingBlock) and block.items:
                # Use different colors for clipped vs non-clipped drawings
                is_clipped = (
                    block.visible_bbox is not None and block.bbox != block.visible_bbox
                )
                color = "magenta" if is_clipped else "cyan"
                draw_path_items(draw, block.items, scale_x, scale_y, color=color)

                # Draw visible_bbox if it differs from bbox (clipped)
                if block.visible_bbox:
                    vbox = [
                        block.visible_bbox.x0 * scale_x,
                        block.visible_bbox.y0 * scale_y,
                        block.visible_bbox.x1 * scale_x,
                        block.visible_bbox.y1 * scale_y,
                    ]
                    draw.rectangle(vbox, outline="yellow", width=1)

                drawings_rendered += 1
                if is_clipped:
                    clipped_count += 1

        logger.debug(
            "Rendered %d drawing paths on page %d (%d clipped, %d unclipped)",
            drawings_rendered,
            result.page_data.page_number,
            clipped_count,
            drawings_rendered - clipped_count,
        )

    img.save(output_path)
    logger.info("Saved image with bboxes to %s", output_path)
