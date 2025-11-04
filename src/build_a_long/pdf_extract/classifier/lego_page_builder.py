"""
Build a Page (LegoPageElement) from ClassificationResult.

This module bridges the gap between the flat list of classified PageElements
and the structured Page hierarchy. It takes the labels and relationships
discovered during classification and constructs a complete LEGO-specific
Page element with all its structured components.
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
from build_a_long.pdf_extract.extractor.hierarchy import build_hierarchy_from_elements
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
from build_a_long.pdf_extract.extractor.page_elements import Element, Text

logger = logging.getLogger(__name__)


class LegoPageBuilder:
    """Builds a Page (LegoPageElement) from classification results.

    This class takes the flat list of classified elements and constructs
    a complete structured Page using LEGO-specific types.
    """

    def __init__(self, page_data: PageData, result: ClassificationResult):
        """Initialize the hierarchy builder.

        Args:
            page_data: The page data containing all elements
            result: Classification result with labels and relationships
        """
        self.page_data = page_data
        self.result = result
        self.warnings: list[str] = []
        self.unprocessed: list[Element] = []

        # Build spatial hierarchy of raw elements for relationship queries
        self.element_tree = build_hierarchy_from_elements(page_data.elements)

        # Track which elements we've already converted
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
            bbox=self.page_data.bbox,
            page_data=self.page_data,
            page_number=page_number,
            steps=steps,
            parts_lists=parts_lists,
            warnings=self.warnings,
            unprocessed_elements=self.unprocessed,
        )

    def _extract_page_number(self) -> PageNumber | None:
        """Extract the page number element."""
        elements = self.result.get_elements_by_label("page_number")

        if not elements:
            return None

        if len(elements) > 1:
            self.warnings.append(
                f"Found {len(elements)} page_number elements, expected at most 1"
            )

        element = elements[0]
        self.converted.add(id(element))

        if isinstance(element, Text):
            # Extract numeric value using shared extraction logic
            value = extract_page_number_value(element.text)
            if value is not None:
                return PageNumber(bbox=element.bbox, value=value)
            else:
                self.warnings.append(
                    f"Could not parse page number from text: '{element.text}'"
                )
                return None
        else:
            self.warnings.append(
                f"page_number element is not Text: {type(element).__name__}"
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

        step_number_elements = self.result.get_elements_by_label("step_number")

        for step_elem in step_number_elements:
            step = self._build_step(step_elem)
            if step:
                steps.append(step)

        return steps

    def _build_step(self, step_elem: Element) -> Step | None:
        """Build a Step from a step_number element.

        Args:
            step_elem: The element labeled as step_number

        Returns:
            A Step object or None if it couldn't be built
        """
        # Extract step number value
        if not isinstance(step_elem, Text):
            self.warnings.append(
                f"step_number element is not Text: {type(step_elem).__name__}"
            )
            return None

        # Extract numeric value using shared extraction logic
        value = extract_step_number_value(step_elem.text)
        if value is None:
            self.warnings.append(
                f"Could not parse step number from text: '{step_elem.text}'"
            )
            return None

        step_number = StepNumber(bbox=step_elem.bbox, value=value)
        self.converted.add(id(step_elem))

        # TODO: Find associated parts_list and diagram
        # For now, we'll create a minimal Step with just the step_number
        # and placeholder diagram

        # Create a placeholder diagram using the step number's bbox
        # In the future, we should find the actual diagram element
        diagram = Diagram(bbox=step_elem.bbox)

        # Create a minimal parts list
        parts_list = PartsList(bbox=step_elem.bbox, parts=[])

        return Step(
            bbox=step_elem.bbox,
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

        parts_list_elements = self.result.get_elements_by_label("parts_list")

        for pl_elem in parts_list_elements:
            # Skip if already processed as part of a step
            if id(pl_elem) in self.converted:
                continue

            parts_list = self._build_parts_list(pl_elem)
            if parts_list:
                parts_lists.append(parts_list)

        return parts_lists

    def _build_parts_list(self, parts_list_elem: Element) -> PartsList | None:
        """Build a PartsList from a parts_list element.

        Args:
            parts_list_elem: The element labeled as parts_list

        Returns:
            A PartsList object or None if it couldn't be built
        """
        self.converted.add(id(parts_list_elem))

        # Find all part_image elements inside this parts_list
        parts = self._extract_parts_from_list(parts_list_elem)

        return PartsList(
            bbox=parts_list_elem.bbox,
            parts=parts,
        )

    def _extract_parts_from_list(self, parts_list_elem: Element) -> list[Part]:
        """Extract Part elements from within a parts_list.

        Uses the part_image_pairs from the classification result to build
        Part objects with their associated PartCount.

        Args:
            parts_list_elem: The parts_list container element

        Returns:
            List of Part objects
        """
        parts: list[Part] = []

        # Get all part_image_pairs
        for part_count_elem, image_elem in self.result.part_image_pairs:
            # Check if this pair is inside the parts_list
            if not self._is_inside(part_count_elem, parts_list_elem):
                continue
            if not self._is_inside(image_elem, parts_list_elem):
                continue

            part = self._build_part(part_count_elem, image_elem)
            if part:
                parts.append(part)
                self.converted.add(id(part_count_elem))
                self.converted.add(id(image_elem))

        return parts

    def _build_part(self, part_count_elem: Element, image_elem: Element) -> Part | None:
        """Build a Part from a part_count and image pair.

        Args:
            part_count_elem: The element labeled as part_count
            image_elem: The element labeled as part_image

        Returns:
            A Part object or None if it couldn't be built
        """
        # Extract count value
        if not isinstance(part_count_elem, Text):
            self.warnings.append(
                f"part_count element is not Text: {type(part_count_elem).__name__}"
            )
            return None

        # Extract numeric value using shared extraction logic
        count_value = extract_part_count_value(part_count_elem.text)
        if count_value is None:
            self.warnings.append(
                f"Could not parse part count from text: '{part_count_elem.text}'"
            )
            return None

        part_count = PartCount(
            bbox=part_count_elem.bbox,
            count=count_value,
        )

        # Combine bboxes of part_count and image to get Part bbox
        combined_bbox = BBox(
            x0=min(part_count_elem.bbox.x0, image_elem.bbox.x0),
            y0=min(part_count_elem.bbox.y0, image_elem.bbox.y0),
            x1=max(part_count_elem.bbox.x1, image_elem.bbox.x1),
            y1=max(part_count_elem.bbox.y1, image_elem.bbox.y1),
        )

        # TODO: Extract part name and number from nearby text elements
        return Part(
            bbox=combined_bbox,
            name=None,
            number=None,
            count=part_count,
        )

    def _is_inside(self, element: Element, container: Element) -> bool:
        """Check if an element is spatially inside a container.

        Args:
            element: The element to check
            container: The potential container element

        Returns:
            True if element is inside container
        """
        return element.bbox.fully_inside(container.bbox)

    def _collect_unprocessed_elements(self) -> None:
        """Collect elements that were classified but not converted."""
        for element in self.page_data.elements:
            # Skip removed elements
            if self.result.is_removed(element):
                continue

            # Skip unlabeled elements
            if not self.result.get_label(element):
                continue

            # Skip already converted elements
            if id(element) in self.converted:
                continue

            self.unprocessed.append(element)


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
    builder = LegoPageBuilder(page_data, result)
    return builder.build()
