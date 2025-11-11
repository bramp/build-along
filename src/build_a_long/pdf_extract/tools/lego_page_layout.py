"""
Generate a diagram showing the layout of LEGO page elements.

This script creates a visual representation of how lego_page_elements are
structured and laid out on a typical LEGO instruction page. It accepts a Page
object and renders it visually, showing the hierarchical structure and spatial
arrangement of all elements.

Run with: pants run src/build_a_long/pdf_extract/tools/lego_page_layout.py
Output: lego_page_layout.png (in src/build_a_long/pdf_extract/)
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    Page,
    PageNumber,
    Part,
    PartCount,
    PartNumber,
    PartsList,
    ProgressBar,
    Step,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing

# Color scheme for different element types
COLORS = {
    "Page": "black",
    "Step": "#4169E1",  # Royal blue
    "PartsList": "#32CD32",  # Lime green
    "Part": "#98FB98",  # Pale green
    "Drawing": "#87CEEB",  # Sky blue
    "PartCount": "#FFD700",  # Gold
    "StepNumber": "#FF6347",  # Tomato red
    "PageNumber": "#8B4513",  # Saddle brown
    "Diagram": "#9370DB",  # Medium purple
    "PartNumber": "#FFA500",  # Orange
    "ProgressBar": "#DDA0DD",  # Plum
}


def draw_lego_brick(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    width: float,
    height: float,
    studs: int = 2,
    color: str = "#E3000F",
) -> None:
    """Draw a simplified LEGO brick with studs."""
    # Draw brick body
    draw.rectangle((x, y, x + width, y + height), fill=color, outline="black", width=1)

    # Draw studs (circles on top of brick)
    stud_radius = 3
    stud_spacing = width // (studs + 1)
    for i in range(studs):
        stud_x = x + stud_spacing * (i + 1)
        stud_y = y + height // 2
        draw.ellipse(
            (
                stud_x - stud_radius,
                stud_y - stud_radius,
                stud_x + stud_radius,
                stud_y + stud_radius,
            ),
            fill="white",
            outline="black",
            width=1,
        )


def draw_bbox_with_label(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    label: str,
    color: str,
    internal_text: str = "",
) -> None:
    """Draw a bounding box with label and optional internal text."""
    x0, y0, x1, y1 = bbox

    # Draw rectangle
    draw.rectangle(bbox, outline=color, width=2)

    # Draw label
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        internal_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()
        internal_font = ImageFont.load_default()

    # Label background (outside, above the box)
    text_bbox = draw.textbbox((x0, y0), label, font=font)
    text_bg = (x0, y0 - 20, text_bbox[2] + 4, y0)
    draw.rectangle(text_bg, fill=color)
    draw.text((x0 + 2, y0 - 18), label, fill="white", font=font)

    # Draw internal text if provided
    if internal_text:
        # Center the text inside the bounding box
        text_bbox = draw.textbbox((0, 0), internal_text, font=internal_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x0 + (x1 - x0 - text_width) // 2
        text_y = y0 + (y1 - y0 - text_height) // 2
        draw.text((text_x, text_y), internal_text, fill=color, font=internal_font)


def draw_page(page: Page, output_path: str = "lego_page_layout.png") -> None:
    """Draw a Page object as a visual diagram.

    Args:
        page: The Page object to render
        output_path: Where to save the output PNG file
    """
    # Use the page's bbox to determine image size
    # Add extra space at the bottom for labels that appear below elements
    page_width = int(page.bbox.x1) + 20
    page_height = int(page.bbox.y1) + 30

    # Create image with white background
    img = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(img, "RGBA")

    # Title
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except OSError:
        title_font = ImageFont.load_default()

    draw.text((10, 10), "LEGO Page Elements Layout", fill="black", font=title_font)

    # Draw page boundary
    page_bbox = (
        int(page.bbox.x0),
        int(page.bbox.y0),
        int(page.bbox.x1),
        int(page.bbox.y1),
    )
    draw.rectangle(page_bbox, outline=COLORS["Page"], width=3)

    # Draw page number if present
    if page.page_number:
        draw_page_number(draw, page.page_number)

    # Draw progress bar if present
    if page.progress_bar:
        draw_progress_bar(draw, page.progress_bar)

    # Draw all steps
    for step in page.steps:
        draw_step(draw, step)

    # Save the image
    img.save(output_path)
    print(f"Diagram saved to {output_path}")


def draw_page_number(draw: ImageDraw.ImageDraw, page_number: PageNumber) -> None:
    """Draw a PageNumber element."""
    bbox = page_number.bbox.to_tuple()
    draw_bbox_with_label(
        draw, bbox, "PageNumber", COLORS["PageNumber"], str(page_number.value)
    )


def draw_progress_bar(draw: ImageDraw.ImageDraw, progress_bar: ProgressBar) -> None:
    """Draw a ProgressBar element."""
    bbox = progress_bar.bbox.to_tuple()
    progress_text = (
        f"{progress_bar.progress:.0%}" if progress_bar.progress is not None else ""
    )
    draw_bbox_with_label(
        draw, bbox, "ProgressBar", COLORS["ProgressBar"], progress_text
    )


def draw_step(draw: ImageDraw.ImageDraw, step: Step) -> None:
    """Draw a Step element with all its children."""
    # Draw step boundary
    bbox = step.bbox.to_tuple()
    draw_bbox_with_label(draw, bbox, "Step", COLORS["Step"])

    # Draw parts list
    draw_parts_list(draw, step.parts_list)

    # Draw step number
    draw_step_number(draw, step.step_number)

    # Draw diagram
    draw_diagram(draw, step.diagram)


def draw_step_number(draw: ImageDraw.ImageDraw, step_number: StepNumber) -> None:
    """Draw a StepNumber element."""
    bbox = step_number.bbox.to_tuple()
    draw_bbox_with_label(
        draw, bbox, "StepNumber", COLORS["StepNumber"], str(step_number.value)
    )


def draw_parts_list(draw: ImageDraw.ImageDraw, parts_list: PartsList) -> None:
    """Draw a PartsList element with all parts."""
    bbox = parts_list.bbox.to_tuple()
    draw_bbox_with_label(draw, bbox, "PartsList", COLORS["PartsList"])

    # Draw each part
    for part in parts_list.parts:
        draw_part(draw, part)


def draw_part(draw: ImageDraw.ImageDraw, part: Part) -> None:
    """Draw a Part element with its drawing and count."""
    # Draw part boundary
    bbox = part.bbox.to_tuple()
    draw_bbox_with_label(draw, bbox, "Part", COLORS["Part"])

    # Draw the part's diagram/drawing if present
    if part.diagram:
        drawing_bbox = part.diagram.bbox.to_tuple()
        draw_bbox_with_label(draw, drawing_bbox, "Drawing", COLORS["Drawing"])

        # Draw a LEGO brick inside the drawing
        x0, y0, x1, y1 = drawing_bbox
        brick_width = min(60, (x1 - x0) - 10)
        brick_height = 20
        brick_x = x0 + ((x1 - x0) - brick_width) // 2
        brick_y = y0 + ((y1 - y0) - brick_height) // 2

        # Vary the brick color and studs
        colors = ["#E3000F", "#0055BF", "#FFC800", "#00852B"]
        brick_color = colors[hash(str(part.bbox)) % len(colors)]
        studs = 2 if (x1 - x0) < 60 else 4

        draw_lego_brick(
            draw, brick_x, brick_y, brick_width, brick_height, studs, brick_color
        )

    # Draw part count
    draw_part_count(draw, part.count)

    # Draw part number if present
    if part.number:
        draw_part_number(draw, part.number)


def draw_part_count(draw: ImageDraw.ImageDraw, part_count: PartCount) -> None:
    """Draw a PartCount element."""
    bbox = part_count.bbox.to_tuple()
    draw_bbox_with_label(
        draw, bbox, "PartCount", COLORS["PartCount"], f"{part_count.count}x"
    )


def draw_part_number(draw: ImageDraw.ImageDraw, part_number: PartNumber) -> None:
    """Draw a PartNumber element."""
    bbox = part_number.bbox.to_tuple()
    draw_bbox_with_label(
        draw, bbox, "PartNumber", COLORS["PartNumber"], str(part_number.element_id)
    )


def draw_diagram(draw: ImageDraw.ImageDraw, diagram: Diagram) -> None:
    """Draw a Diagram element."""
    bbox = diagram.bbox.to_tuple()
    draw_bbox_with_label(draw, bbox, "Diagram", COLORS["Diagram"])

    # Add placeholder text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()

    x0, y0, x1, y1 = bbox
    center_x = x0 + (x1 - x0) // 2
    center_y = y0 + (y1 - y0) // 2

    draw.text(
        (center_x - 60, center_y - 20),
        "Assembly\nInstruction\nGraphic",
        fill=COLORS["Diagram"],
        font=font,
        align="center",
    )


def create_sample_page() -> Page:
    """Create a sample Page for demonstration purposes.

    This creates a representative page with 2 steps showing typical layout.
    All coordinates are hardcoded for clarity.
    """
    return Page(
        bbox=BBox(40, 50, 760, 620),
        page_number=PageNumber(
            value=10,
            bbox=BBox(680, 570, 720, 595),
        ),
        progress_bar=ProgressBar(
            bbox=BBox(60, 570, 670, 595),
            progress=0.25,
        ),
        steps=[
            # Step 1
            Step(
                bbox=BBox(60, 100, 740, 350),
                step_number=StepNumber(
                    value=1,
                    bbox=BBox(70, 210, 110, 240),
                ),
                parts_list=PartsList(
                    bbox=BBox(70, 110, 290, 200),
                    parts=[
                        # Part 1
                        Part(
                            bbox=BBox(80, 120, 140, 185),
                            count=PartCount(
                                count=2,
                                bbox=BBox(85, 165, 110, 180),
                            ),
                            diagram=Drawing(
                                id=1,
                                bbox=BBox(85, 125, 135, 160),
                            ),
                        ),
                        # Part 2
                        Part(
                            bbox=BBox(150, 120, 280, 185),
                            count=PartCount(
                                count=1,
                                bbox=BBox(155, 165, 185, 180),
                            ),
                            diagram=Drawing(
                                id=2,
                                bbox=BBox(155, 125, 225, 160),
                            ),
                            number=PartNumber(
                                element_id="6091234",
                                bbox=BBox(230, 125, 275, 145),
                            ),
                        ),
                    ],
                ),
                diagram=Diagram(
                    bbox=BBox(310, 110, 730, 340),
                ),
            ),
            # Step 2
            Step(
                bbox=BBox(60, 380, 740, 540),
                step_number=StepNumber(
                    value=2,
                    bbox=BBox(70, 490, 110, 520),
                ),
                parts_list=PartsList(
                    bbox=BBox(70, 390, 220, 480),
                    parts=[
                        # Part 4
                        Part(
                            bbox=BBox(80, 400, 180, 475),
                            count=PartCount(
                                count=2,
                                bbox=BBox(85, 455, 115, 470),
                            ),
                            diagram=Drawing(
                                id=4,
                                bbox=BBox(85, 405, 175, 450),
                            ),
                        ),
                    ],
                ),
                diagram=Diagram(
                    bbox=BBox(240, 390, 730, 530),
                ),
            ),
        ],
    )


if __name__ == "__main__":
    # Create a sample page showing typical layout
    sample_page = create_sample_page()

    # Output to the pdf_extract directory
    # This path works both when run via pants and directly
    output_path = "src/build_a_long/pdf_extract/lego_page_layout.png"

    draw_page(sample_page, output_path)
