import logging
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from build_a_long.pdf_extract.classifier import ClassificationResult
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


class DrawableItem(BaseModel):
    """A unified structure for things to draw on the page."""

    model_config = ConfigDict(frozen=True)

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

    is_unassigned: bool = False
    """True if this block has no candidates."""

    depth: int = 0
    """Nesting depth for color selection (computed later)."""


def _create_drawable_items(
    result: ClassificationResult,
    *,
    draw_blocks: bool,
    draw_elements: bool,
    draw_deleted: bool,
    draw_unassigned: bool = False,
    debug_candidates_label: str | None = None,
) -> list[DrawableItem]:
    """Create a unified list of items to draw.

    Args:
        result: Classification result containing blocks and elements
        draw_blocks: If True, include PDF blocks
        draw_elements: If True, include LEGO page elements
        draw_deleted: If True, include removed/non-winner items
        draw_unassigned: If True, include blocks with no candidates
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

                # TODO I wonder if we should draw a dashed box for the original
                # box if the constructed element bbox is different?

                # Use constructed element's bbox if available (it may have been
                # updated after construction, e.g., Step bbox includes diagram)
                # Otherwise fall back to candidate bbox
                bbox = (
                    candidate.constructed.bbox
                    if candidate.constructed
                    else candidate.bbox
                )

                # Element without source block (e.g., Step, Page)
                items.append(
                    DrawableItem(
                        bbox=bbox,
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

        # Also add elements from the Page hierarchy that don't have candidates
        # (e.g., SubAssemblyStep elements created as substeps without a candidate)
        if result.page:
            drawn_element_ids = {
                id(c.constructed)
                for _, cs in all_candidates.items()
                for c in cs
                if c.constructed
            }
            for element in result.page.iter_elements():
                if id(element) in drawn_element_ids:
                    continue
                if id(element) not in chosen_elements:
                    continue
                # This element is in the page hierarchy but wasn't drawn via a candidate
                label = f"[{element.__class__.__name__}]"
                items.append(
                    DrawableItem(
                        bbox=element.bbox,
                        label=label,
                        is_element=True,
                        is_winner=True,
                        is_removed=False,
                    )
                )

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

    # Add unassigned blocks (blocks with no candidates)
    if draw_unassigned:
        for block in result.blocks:
            # Skip blocks that are source blocks for elements
            if block.id in element_source_block_ids:
                continue

            # Skip removed blocks
            if result.is_removed(block):
                continue

            # Check if this block has any candidates
            candidates = result.get_all_candidates_for_block(block)
            if candidates:
                continue

            # This is an unassigned block - no candidates at all
            label = f"ID: {block.id} [UNASSIGNED] {str(block)}"
            items.append(
                DrawableItem(
                    bbox=block.bbox,
                    label=label,
                    is_element=False,
                    is_winner=False,
                    is_removed=False,
                    is_unassigned=True,
                )
            )

    return items


def _draw_item(
    draw: ImageDraw.ImageDraw,
    item: DrawableItem,
    depth_colors: list[str],
    scale_x: float,
    scale_y: float,
    image_width: int,
    image_height: int,
) -> None:
    """Draw a single item on the image.

    Args:
        draw: PIL ImageDraw object
        item: The item to draw
        depth_colors: List of colors to cycle through
        scale_x: X scaling factor
        scale_y: Y scaling factor
        image_width: Width of the image
        image_height: Height of the image
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
    if item.is_unassigned:
        # Unassigned blocks get solid magenta with thick lines
        draw.rectangle(scaled_bbox, outline="magenta", width=3)
        color = "magenta"  # Use magenta for label too
    elif item.is_element:
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

    # Calculate text size
    text_bbox = draw.textbbox((0, 0), item.label)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Determine text position
    # Left aligned if depth is even, Right aligned if odd
    text_x = scaled_bbox[0] if item.depth % 2 == 0 else scaled_bbox[2] - text_width

    # Default to drawing below the box
    text_y = scaled_bbox[3] + 2

    # Check if text is off-screen (vertically)
    if text_y + text_height > image_height:
        # Move text inside the box (aligned to bottom)
        text_y = scaled_bbox[3] - text_height - 2

    draw.text((text_x, text_y), item.label, fill=color)


def draw_and_save_bboxes(
    page: pymupdf.Page,
    result: ClassificationResult,
    output_path: Path,
    *,
    draw_blocks: bool = False,
    draw_elements: bool = False,
    draw_deleted: bool = False,
    draw_drawings: bool = False,
    draw_unassigned: bool = False,
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
        draw_unassigned: If True, render blocks with no candidates.
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
        draw_unassigned=draw_unassigned,
        debug_candidates_label=debug_candidates_label,
    )

    # Build hierarchy for depth calculation directly from DrawableItems
    hierarchy = build_hierarchy_from_blocks(items)

    # Compute depth and create new items with depth set (since items are frozen)
    items_with_depth = []
    for item in items:
        depth = hierarchy.get_depth(item)
        # Create new item with depth
        items_with_depth.append(
            DrawableItem(
                bbox=item.bbox,
                label=item.label,
                is_element=item.is_element,
                is_winner=item.is_winner,
                is_removed=item.is_removed,
                is_unassigned=item.is_unassigned,
                depth=depth,
            )
        )

    # Draw all items
    for item in items_with_depth:
        _draw_item(
            draw,
            item,
            depth_colors,
            scale_x,
            scale_y,
            image_width=pix.width,
            image_height=pix.height,
        )

    # Draw actual drawing paths if requested
    if draw_drawings:
        drawings_rendered = 0
        clipped_count = 0
        for block in result.page_data.blocks:
            if isinstance(block, DrawingBlock) and block.items:
                # Use different colors for clipped vs non-clipped drawings
                is_clipped = block.is_clipped
                color = "magenta" if is_clipped else "cyan"
                draw_path_items(draw, block.items, scale_x, scale_y, color=color)

                # Draw visible bbox (which is now just bbox) in yellow
                vbox = [
                    block.bbox.x0 * scale_x,
                    block.bbox.y0 * scale_y,
                    block.bbox.x1 * scale_x,
                    block.bbox.y1 * scale_y,
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
