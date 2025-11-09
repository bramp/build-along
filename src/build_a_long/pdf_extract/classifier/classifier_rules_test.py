"""Rule-based tests over real fixtures for the PDF element classifier.

This suite validates high-level invariants that must hold after classification.

Rules covered:
- Every parts list must contain at least one part image inside it.
- No two parts lists overlap.
- Each part image is inside a parts list.
- Each element has at most one winner candidate.

Real fixture(s) live under this package's fixtures/ directory.
"""

import logging
from collections import defaultdict
from pathlib import Path

import pytest

from build_a_long.pdf_extract.classifier import ClassificationResult, classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Block, Text

log = logging.getLogger(__name__)

# TODO A lot of the methods in ClassifiedPage overlap with ClassificationResult


class ClassifiedPage:
    """Wrapper around PageData providing convenient access to classified elements.

    This class provides helper methods to query elements by label type and
    supports hierarchical queries (e.g., finding children inside parent bboxes).
    Results are cached for efficiency.
    """

    def __init__(self, page: PageData, result: ClassificationResult):
        """Initialize with a classified PageData and its result.

        Args:
            page: PageData that has been run through classify_elements()
            result: The ClassificationResult for this page
        """
        self.page = page
        self.result = result
        self._cache: dict[str, list[Block]] = {}

    def elements_by_label(
        self, label: str, include_deleted: bool = False
    ) -> list[Block]:
        """Get all elements with the given label.

        Args:
            label: The label to filter by
            include_deleted: Whether to include deleted elements

        Returns:
            List of elements with matching label
        """
        cache_key = f"{label}:deleted={include_deleted}"
        if cache_key not in self._cache:
            if include_deleted:
                self._cache[cache_key] = [
                    e for e in self.page.blocks if self.result.get_label(e) == label
                ]
            else:
                self._cache[cache_key] = [
                    e
                    for e in self.page.blocks
                    if self.result.get_label(e) == label
                    and not self.result.is_removed(e)
                ]
        return self._cache[cache_key]

    def parts_lists(self) -> list[Block]:
        """Get all non-deleted parts_list elements."""
        return self.elements_by_label("parts_list")

    def part_images(self) -> list[Block]:
        """Get all non-deleted part_image elements."""
        return self.elements_by_label("part_image")

    def part_counts(self) -> list[Block]:
        """Get all non-deleted part_count elements."""
        return self.elements_by_label("part_count")

    def step_numbers(self) -> list[Block]:
        """Get all non-deleted step_number elements."""
        return self.elements_by_label("step_number")

    def children_of(self, parent: Block, label: str | None = None) -> list[Block]:
        """Return all non-deleted elements spatially contained within a parent element.

        Note: This uses bbox containment, not ElementTree hierarchy, because the hierarchy
        is based on "smallest containing bbox" which means there may be intermediate
        unlabeled elements between a parent and its logical children. For validation
        rules about spatial containment, bbox checking is more appropriate.

        Args:
            parent: The parent element to search within
            label: Optional label filter (e.g., "part_image")

        Returns:
            List of non-deleted Elements matching the label (if specified) that
            are fully contained within the parent's bbox
        """
        # Use spatial containment, not hierarchy
        result = []
        for elem in self.page.blocks:
            if id(elem) in self.result.removal_reasons:
                continue
            if label is not None and self.result.get_label(elem) != label:
                continue
            if elem.bbox.fully_inside(parent.bbox):
                result.append(elem)
        return result

    def print_summary(self, logger: logging.Logger | None = None) -> None:
        """Log a summary of labeled elements.

        Args:
            logger: Logger to use (defaults to module logger)
        """
        logger = logger or log
        label_counts = defaultdict(int)
        for e in self.page.blocks:
            label = (
                self.result.get_label(e) if self.result.get_label(e) else "<unknown>"
            )
            label_counts[label] += 1

        logger.info(f"Label counts: {dict(label_counts)}")


# TODO Replace this with just results.get_blocks_by_label()


def _parts_lists(page: PageData, result: ClassificationResult) -> list[Block]:
    return [
        e
        for e in page.blocks
        if result.get_label(e) == "parts_list" and not result.is_removed(e)
    ]


# TODO Replace this with just results.get_blocks_by_label()


def _part_images(page: PageData, result: ClassificationResult) -> list[Block]:
    return [
        e
        for e in page.blocks
        if result.get_label(e) == "part_image" and not result.is_removed(e)
    ]


# TODO Replace this with just results.get_blocks_by_label()


def _part_counts(page: PageData, result: ClassificationResult) -> list[Block]:
    return [
        e
        for e in page.blocks
        if result.get_label(e) == "part_count" and not result.is_removed(e)
    ]


def _print_label_counts(page: PageData, result: ClassificationResult) -> None:
    label_counts = defaultdict(int)
    for e in page.blocks:
        label = result.get_label(e) if result.get_label(e) else "<unknown>"
        label_counts[label] += 1

    # TODO The following logging shows "defaultdict(<class 'int'>,..." figure
    # out how to avoid that.
    log.info(f"Label counts: {label_counts}")


@pytest.mark.skip(reason="Not working yet.")
class TestClassifierRules:
    """End-to-end rules that must hold on real pages after classification."""

    @pytest.mark.parametrize(
        "fixture_file",
        [f.name for f in (Path(__file__).parent.parent / "fixtures").glob("*.json")],
    )
    def test_parts_list_contains_at_least_one_part_image(
        self, fixture_file: str
    ) -> None:
        """Every labeled parts list should include at least one part image inside its bbox.

        This test runs on all JSON fixtures in the fixtures/ directory.
        """

        fixture_path = Path(__file__).parent.parent / "fixtures" / fixture_file
        page: PageData = PageData.model_validate_json(fixture_path.read_text())

        # Run the full classification pipeline on the page
        result = classify_elements(page)

        classified = ClassifiedPage(page, result)
        classified.print_summary()

        parts_lists = classified.parts_lists()
        part_images = classified.part_images()
        part_counts = classified.part_counts()

        # Debug: show all part_image labeled elements including deleted ones
        all_part_images = classified.elements_by_label(
            "part_image", include_deleted=True
        )
        log.info(
            f"Total on page: {len(parts_lists)} parts_lists, {len(part_images)} part_images (non-deleted), {len(all_part_images)} total part_images, {len(part_counts)} part_counts"
        )
        if len(all_part_images) != len(part_images):
            deleted_count = len(all_part_images) - len(part_images)
            log.warning(
                f"  WARNING: {deleted_count} part_images are DELETED on this page"
            )
            for img in all_part_images:
                if result.is_removed(img):
                    # Check if it's inside any parts_list
                    inside_any = any(
                        img.bbox.fully_inside(pl.bbox) for pl in parts_lists
                    )
                    location = (
                        "inside a parts_list"
                        if inside_any
                        else "outside all parts_lists"
                    )
                    log.warning(
                        f"    - Deleted PartImage id:{img.id} bbox:{img.bbox} ({location})"
                    )

        for parts_list in parts_lists:
            part_images_inside = classified.children_of(parts_list, label="part_image")
            part_counts_inside = classified.children_of(parts_list, label="part_count")

            # Also get ALL part_images (including deleted) to check for deletion bugs
            all_part_images_inside = []
            for elem in page.blocks:
                if result.get_label(elem) == "part_image" and elem.bbox.fully_inside(
                    parts_list.bbox
                ):
                    all_part_images_inside.append(elem)

            log.info(
                f"{fixture_file} PartsList id:{parts_list.id} bbox:{parts_list.bbox} contains:"
            )
            for img in part_images_inside:
                log.info(f" - PartImage id:{img.id} bbox:{img.bbox}")
            for count in part_counts_inside:
                count_text = count.text if isinstance(count, Text) else ""
                log.info(
                    f" - PartCount id:{count.id} text:{count_text} bbox:{count.bbox}"
                )

            # Log deleted part_images if any
            deleted_images = [
                img for img in all_part_images_inside if result.is_removed(img)
            ]
            if deleted_images:
                log.warning(
                    f"  WARNING: {len(deleted_images)} part_images DELETED inside parts_list {parts_list.id}:"
                )
                for img in deleted_images:
                    log.warning(
                        f"    - PartImage id:{img.id} bbox:{img.bbox} [DELETED]"
                    )

            # Debug: log all part images to see why they're not inside
            if len(part_images_inside) == 0:
                log.info("  DEBUG: All part_images on page:")
                for img in part_images:
                    log.info(
                        f"  - PartImage id:{img.id} bbox:{img.bbox} inside:{img.bbox.fully_inside(parts_list.bbox)}"
                    )

            # Each parts_list must contain at least one part_image fully inside its bbox
            assert len(part_images_inside) >= 1, (
                f"Parts list {parts_list.id} in {fixture_file} should contain at least one part image"
            )

            # No part_images inside a parts_list should be deleted
            assert len(deleted_images) == 0, (
                f"Parts list {parts_list.id} in {fixture_file} has {len(deleted_images)} "
                f"deleted part_images inside it (should be 0)"
            )

            # Each parts_list must contain the same number of part_counts as
            # part_images inside it
            assert len(part_counts_inside) == len(part_images_inside), (
                f"PartsList id:{parts_list.id} in {fixture_file} should contain "
                f"{len(part_images_inside)} PartCounts, found {len(part_counts_inside)}"
            )

    @pytest.mark.parametrize(
        "fixture_file",
        [f.name for f in (Path(__file__).parent.parent / "fixtures").glob("*.json")],
    )
    def test_parts_lists_do_not_overlap(self, fixture_file: str) -> None:
        """No two parts lists should overlap.

        Parts lists represent distinct areas of the page and should not
        have overlapping bounding boxes.
        """
        fixture_path = Path(__file__).parent.parent / "fixtures" / fixture_file
        page: PageData = PageData.model_validate_json(fixture_path.read_text())

        # Run the full classification pipeline on the page
        result = classify_elements(page)

        classified = ClassifiedPage(page, result)
        parts_lists = classified.parts_lists()

        # Check all pairs of parts lists for overlap
        for i, parts_list_a in enumerate(parts_lists):
            for parts_list_b in parts_lists[i + 1 :]:
                assert not parts_list_a.bbox.overlaps(parts_list_b.bbox), (
                    f"Parts lists {parts_list_a.id} (bbox:{parts_list_a.bbox}) and "
                    f"{parts_list_b.id} (bbox:{parts_list_b.bbox}) in {fixture_file} overlap"
                )

    @pytest.mark.parametrize(
        "fixture_file",
        [f.name for f in (Path(__file__).parent.parent / "fixtures").glob("*.json")],
    )
    def test_each_part_image_is_inside_a_parts_list(self, fixture_file: str) -> None:
        """Each part image must be inside at least one parts list.

        Every part_image should be contained within a parts_list's bounding box.
        """
        fixture_path = Path(__file__).parent.parent / "fixtures" / fixture_file
        page: PageData = PageData.model_validate_json(fixture_path.read_text())

        # Run the full classification pipeline on the page
        result = classify_elements(page)

        classified = ClassifiedPage(page, result)
        parts_lists = classified.parts_lists()
        part_images = classified.part_images()

        for part_image in part_images:
            # Check if this part_image is inside at least one parts_list
            inside_any_parts_list = any(
                part_image.bbox.fully_inside(pl.bbox) for pl in parts_lists
            )

            assert inside_any_parts_list, (
                f"Part image {part_image.id} (bbox:{part_image.bbox}) in {fixture_file} "
                f"is not inside any parts_list"
            )

    @pytest.mark.parametrize(
        "fixture_file",
        [f.name for f in (Path(__file__).parent.parent / "fixtures").glob("*.json")],
    )
    def test_no_labeled_element_is_deleted(self, fixture_file: str) -> None:
        """No element with a label should be marked as deleted.

        If an element has been classified with a label, it should not be deleted.
        This ensures that the classification and deletion logic don't conflict.
        """
        fixture_path = Path(__file__).parent.parent / "fixtures" / fixture_file
        page: PageData = PageData.model_validate_json(fixture_path.read_text())

        # Run the full classification pipeline on the page
        result = classify_elements(page)

        # Find all elements that are both labeled and deleted
        labeled_and_deleted = []
        for elem in page.blocks:
            if result.get_label(elem) is not None and result.is_removed(elem):
                labeled_and_deleted.append(elem)

        if labeled_and_deleted:
            log.error(
                f"Found {len(labeled_and_deleted)} labeled elements that are deleted:"
            )
            for elem in labeled_and_deleted:
                log.error(
                    f"  - {result.get_label(elem)} id:{elem.id} bbox:{elem.bbox} [DELETED]"
                )

        assert len(labeled_and_deleted) == 0, (
            f"Found {len(labeled_and_deleted)} labeled elements that are deleted in {fixture_file}. "
            f"Labeled elements should not be deleted."
        )

    @pytest.mark.parametrize(
        "fixture_file",
        [f.name for f in (Path(__file__).parent.parent / "fixtures").glob("*.json")],
    )
    def test_each_element_has_at_most_one_winner(self, fixture_file: str) -> None:
        """Each element should have at most one winner candidate across all labels.

        An element can have multiple candidates across different labels, but only
        one of them should be marked as a winner. This ensures classification
        decisions are unambiguous.
        """
        fixture_path = Path(__file__).parent.parent / "fixtures" / fixture_file
        page: PageData = PageData.model_validate_json(fixture_path.read_text())

        # Run the full classification pipeline on the page
        result = classify_elements(page)

        # Track which blocks have won, and for which label
        block_to_winning_label: dict[int, str] = {}

        # Check all candidates across all labels
        all_candidates = result.get_all_candidates()
        for label, candidates in all_candidates.items():
            for candidate in candidates:
                if not candidate.is_winner:
                    continue

                # Skip synthetic candidates (no source block)
                if candidate.source_block is None:
                    continue

                block_id = candidate.source_block.id

                # Check if this block already has a winner
                if block_id in block_to_winning_label:
                    existing_label = block_to_winning_label[block_id]
                    pytest.fail(
                        f"Block {block_id} in {fixture_file} has multiple winner candidates: "
                        f"'{existing_label}' and '{label}'. Each block should have at most one winner."
                    )

                block_to_winning_label[block_id] = label
