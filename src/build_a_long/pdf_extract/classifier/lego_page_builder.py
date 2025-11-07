"""
Build a Page (LegoPageElement) from ClassificationResult.

This module bridges the gap between the flat list of classified Blocks
(raw PDF primitives) and the structured Page hierarchy. It takes the labels
and relationships discovered during classification and constructs a complete
LEGO-specific Page element with all its structured components.
"""

from __future__ import annotations

import logging

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.text_extractors import (
    extract_page_number_value,
    extract_part_count_value,
    extract_step_number_value,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.hierarchy import build_hierarchy_from_blocks
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    Page,
    PageNumber,
    Part,
    PartCount,
    PartsList,
    Step,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Block, Text

logger = logging.getLogger(__name__)


class LegoPageBuilder:
    """Builds a Page (LegoPageElement) from classification results.

    This class takes the flat list of classified blocks and constructs
    a complete structured Page using LEGO-specific types.
    """

    def __init__(self, result: ClassificationResult):
        """Initialize the hierarchy builder.

        Args:
            result: Classification result with labels and relationships
        """
        self.result = result
        self.warnings: list[str] = []
        self.unprocessed: list[Block] = []

        # Build spatial hierarchy of raw blocks for relationship queries
        self.block_tree = build_hierarchy_from_blocks(self.result.page_data.blocks)

        # Track which blocks we've already converted
        self.converted: set[int] = set()

    def build(self) -> Page:
        """Build the Page with all structured LEGO elements.

        Returns:
            Page containing the structured elements
        """
        # Extract page number
        page_number = self._extract_page_number()

        # Extract steps (which may contain parts lists)
        steps = self._extract_steps()

        # Extract standalone parts lists (not inside steps)
        parts_lists = self._extract_standalone_parts_lists()

        # Collect any elements that were classified but not converted
        self._collect_unprocessed_elements()

        # Use the page's bbox from the original page data
        return Page(
            bbox=self.result.page_data.bbox,
            page_number=page_number,
            steps=steps,
            parts_lists=parts_lists,
            warnings=self.warnings,
            unprocessed_elements=self.unprocessed,
        )

    def _extract_page_number(self) -> PageNumber | None:
        """Extract the page number block."""
        blocks = self.result.get_blocks_by_label("page_number")

        if not blocks:
            return None

        if len(blocks) > 1:
            self.warnings.append(
                f"Found {len(blocks)} page_number blocks, expected at most 1"
            )

        block = blocks[0]
        self.converted.add(id(block))

        if isinstance(block, Text):
            # Extract numeric value using shared extraction logic
            value = extract_page_number_value(block.text)
            if value is not None:
                return PageNumber(bbox=block.bbox, value=value)
            else:
                self.warnings.append(
                    f"Could not parse page number from text: '{block.text}'"
                )
                return None
        else:
            self.warnings.append(
                f"page_number block is not Text: {type(block).__name__}"
            )
            return None

    def _extract_steps(self) -> list[Step]:
        """Extract Step elements from the page.

        A Step consists of:
        - A StepNumber
        - A PartsList (optional, may be shared across pages)
        - A Diagram (the main instruction graphic)
        """
        steps: list[Step] = []

        step_number_blocks = self.result.get_blocks_by_label("step_number")

        for step_block in step_number_blocks:
            step = self._build_step(step_block)
            if step:
                steps.append(step)

        return steps

    def _build_step(self, step_block: Block) -> Step | None:
        """Build a Step from a step_number block.

        Args:
            step_block: The block labeled as step_number

        Returns:
            A Step object or None if it couldn't be built
        """
        # Extract step number value
        if not isinstance(step_block, Text):
            self.warnings.append(
                f"step_number block is not Text: {type(step_block).__name__}"
            )
            return None

        # Extract numeric value using shared extraction logic
        value = extract_step_number_value(step_block.text)
        if value is None:
            self.warnings.append(
                f"Could not parse step number from text: '{step_block.text}'"
            )
            return None

        step_number = StepNumber(bbox=step_block.bbox, value=value)
        self.converted.add(id(step_block))

        # TODO: Find associated parts_list and diagram
        # For now, we'll create a minimal Step with just the step_number
        # and placeholder diagram

        # Create a placeholder diagram using the step number's bbox
        # In the future, we should find the actual diagram block
        diagram = Diagram(bbox=step_block.bbox)

        # Create a minimal parts list
        parts_list = PartsList(bbox=step_block.bbox, parts=[])

        return Step(
            bbox=step_block.bbox,
            step_number=step_number,
            parts_list=parts_list,
            diagram=diagram,
        )

    def _extract_standalone_parts_lists(self) -> list[PartsList]:
        """Extract PartsList elements that are not within a Step.

        Returns:
            List of PartsList objects
        """
        parts_lists: list[PartsList] = []

        parts_list_blocks = self.result.get_blocks_by_label("parts_list")

        for pl_block in parts_list_blocks:
            # Skip if already processed as part of a step
            if id(pl_block) in self.converted:
                continue

            parts_list = self._build_parts_list(pl_block)
            if parts_list:
                parts_lists.append(parts_list)

        return parts_lists

    def _build_parts_list(self, parts_list_block: Block) -> PartsList | None:
        """Build a PartsList from a parts_list block.

        Args:
            parts_list_block: The block labeled as parts_list

        Returns:
            A PartsList object or None if it couldn't be built
        """
        self.converted.add(id(parts_list_block))

        # Find all part_image blocks inside this parts_list
        parts = self._extract_parts_from_list(parts_list_block)

        return PartsList(
            bbox=parts_list_block.bbox,
            parts=parts,
        )

    def _extract_parts_from_list(self, parts_list_block: Block) -> list[Part]:
        """Extract Part elements from within a parts_list.

        Uses the part_image_pairs from the classification result to build
        Part objects with their associated PartCount.

        Args:
            parts_list_block: The parts_list container block

        Returns:
            List of Part objects
        """
        parts: list[Part] = []

        # Get all part_image_pairs
        for part_count_block, image_block in self.result.get_part_image_pairs():
            # Check if this pair is inside the parts_list
            if not self._is_inside(part_count_block, parts_list_block):
                continue
            if not self._is_inside(image_block, parts_list_block):
                continue

            part = self._build_part(part_count_block, image_block)
            if part:
                parts.append(part)
                self.converted.add(id(part_count_block))
                self.converted.add(id(image_block))

        return parts

    def _build_part(self, part_count_block: Block, image_block: Block) -> Part | None:
        """Build a Part from a part_count and image pair.

        Args:
            part_count_block: The block labeled as part_count
            image_block: The block labeled as part_image

        Returns:
            A Part object or None if it couldn't be built
        """
        # Extract count value
        if not isinstance(part_count_block, Text):
            self.warnings.append(
                f"part_count block is not Text: {type(part_count_block).__name__}"
            )
            return None

        # Extract numeric value using shared extraction logic
        count_value = extract_part_count_value(part_count_block.text)
        if count_value is None:
            self.warnings.append(
                f"Could not parse part count from text: '{part_count_block.text}'"
            )
            return None

        part_count = PartCount(
            bbox=part_count_block.bbox,
            count=count_value,
        )

        # Combine bboxes of part_count and image to get Part bbox
        combined_bbox = BBox(
            x0=min(part_count_block.bbox.x0, image_block.bbox.x0),
            y0=min(part_count_block.bbox.y0, image_block.bbox.y0),
            x1=max(part_count_block.bbox.x1, image_block.bbox.x1),
            y1=max(part_count_block.bbox.y1, image_block.bbox.y1),
        )

        # TODO: Extract part name and number from nearby text blocks
        return Part(
            bbox=combined_bbox,
            name=None,
            number=None,
            count=part_count,
        )

    def _is_inside(self, block: Block, container: Block) -> bool:
        """Check if a block is spatially inside a container.

        Args:
            block: The block to check
            container: The potential container block

        Returns:
            True if block is inside container
        """
        return block.bbox.fully_inside(container.bbox)

    def _collect_unprocessed_elements(self) -> None:
        """Collect blocks that were classified but not converted."""
        for block in self.result.page_data.blocks:
            # Skip removed blocks
            if self.result.is_removed(block):
                continue

            # Skip unlabeled blocks
            if not self.result.get_label(block):
                continue

            # Skip already converted blocks
            if id(block) in self.converted:
                continue

            self.unprocessed.append(block)


def build_page(
    page_data: PageData,
    result: ClassificationResult,
) -> Page:
    """Build a Page (LegoPageElement) from classification results.

    This is the main entry point for converting classified page elements
    into a structured LEGO-specific Page.

    Args:
        page_data: The page data containing all elements
        result: Classification result with labels and relationships

    Returns:
        Page element containing the structured hierarchy and any warnings

    Example:
        >>> pages = extract_bounding_boxes(doc, None)
        >>> results = classify_pages(pages)
        >>> for page_data, result in zip(pages, results):
        ...     page = build_page(page_data, result)
        ...     if page.page_number:
        ...         print(f"Page {page.page_number.value}")
        ...     for step in page.steps:
        ...         print(f"  Step {step.step_number.value}")
    """
    builder = LegoPageBuilder(result)
    return builder.build()
