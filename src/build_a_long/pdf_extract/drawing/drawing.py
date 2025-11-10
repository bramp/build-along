import logging
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.extractor.hierarchy import build_hierarchy_from_blocks
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Text,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Image as ImageBlock,
)

logger = logging.getLogger(__name__)


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    outline: str | None = None,
    width: int = 0,
    dash_length: int = 5,
) -> None:
    x0, y0, x1, y1 = bbox
    # Top edge
    for i in range(int(x0), int(x1), dash_length * 2):
        draw.line([(i, y0), (min(i + dash_length, x1), y0)], fill=outline, width=width)
    # Bottom edge
    for i in range(int(x0), int(x1), dash_length * 2):
        draw.line([(i, y1), (min(i + dash_length, x1), y1)], fill=outline, width=width)
    # Left edge
    for i in range(int(y0), int(y1), dash_length * 2):
        draw.line([(x0, i), (x0, min(i + dash_length, y1))], fill=outline, width=width)
    # Right edge
    for i in range(int(y0), int(y1), dash_length * 2):
        draw.line([(x1, i), (x1, min(i + dash_length, y1))], fill=outline, width=width)


def draw_and_save_bboxes(
    page: pymupdf.Page,
    result: ClassificationResult,
    output_path: Path,
    *,
    draw_deleted: bool = False,
) -> None:
    """
    Draws bounding boxes from blocks on the PDF page image and saves it.
    Colors are based on nesting depth (calculated via bbox containment).

    Args:
        page: PyMuPDF page to render
        result: ClassificationResult containing labels and blocks
        output_path: Where to save the output image
        draw_deleted: If True, also render blocks marked as deleted.
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

    # Build hierarchy once to efficiently calculate depths - O(n log n)
    hierarchy = build_hierarchy_from_blocks(result.blocks)

    # Draw all blocks
    for block in result.blocks:
        block_removed = result.is_removed(block)
        if block_removed and not draw_deleted:
            continue

        bbox = block.bbox

        # Scale the bounding box
        scaled_bbox = (
            bbox.x0 * scale_x,
            bbox.y0 * scale_y,
            bbox.x1 * scale_x,
            bbox.y1 * scale_y,
        )

        # Get pre-calculated depth - O(1)
        depth = hierarchy.get_depth(block)

        # Determine color based on depth
        color = depth_colors[depth % len(depth_colors)]

        # If block is removed (in to_remove), use lighter/grayed color and dash
        if block_removed:
            # Draw dashed outline for removed blocks
            _draw_dashed_rectangle(draw, scaled_bbox, outline=color, width=2)
        else:
            # Draw the bounding box normally
            draw.rectangle(scaled_bbox, outline=color, width=1)

        # Draw the block type text
        label_prefix = "[REMOVED] " if block_removed else ""
        block_label = result.get_label(block)  # type: ignore[arg-type]
        label = f"ID: {block.id} {label_prefix}" + (
            block_label or block.__class__.__name__
        )
        if isinstance(block, Drawing | ImageBlock):
            if block.image_id:
                label = f"{label} ({block.image_id})"
        elif isinstance(block, Text):
            # For Text blocks, show the actual text content
            content = block.text.strip()
            if len(content) > 50:  # Truncate long text
                content = content[:47] + "..."
            label = f"{label}: {content}"

        # Below bottom-left
        if depth % 2 == 0:  # Even depth, left align
            text_position = (scaled_bbox[0], scaled_bbox[3] + 2)
            draw.text(text_position, label, fill=color)
        else:  # Odd depth, right align
            # Calculate text width for right alignment
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_position = (scaled_bbox[2] - text_width, scaled_bbox[3] + 2)
            draw.text(text_position, label, fill=color)

    # Draw all winning candidates that are synthetic (no source_block)
    synthetic_winner_color = "magenta"
    for _, candidates in result.get_all_candidates().items():
        for candidate in candidates:
            if candidate.is_winner and candidate.source_block is None:
                bbox = candidate.bbox

                # Scale the bounding box
                scaled_bbox = (
                    bbox.x0 * scale_x,
                    bbox.y0 * scale_y,
                    bbox.x1 * scale_x,
                    bbox.y1 * scale_y,
                )

                # Draw outline for synthetic winners
                draw.rectangle(scaled_bbox, outline=synthetic_winner_color, width=2)

                # Draw the label and score
                label = f"SYNTHETIC WINNER: {candidate.label} (Score: {candidate.score:.2f})"
                text_position = (scaled_bbox[0], scaled_bbox[1] - 12)  # Above top-left
                draw.text(text_position, label, fill=synthetic_winner_color)

    img.save(output_path)
    logger.info("Saved image with bboxes to %s", output_path)
