"""
Page classifier.

Purpose
-------
Build a complete Page element from classified components.
This classifier depends on page_number and step to construct the final Page.

Debugging
---------
Set environment variables to aid investigation without code changes:

- LOG_LEVEL=DEBUG
    Enables DEBUG-level logging (if not already configured by caller).
"""

import logging
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LegoPageElements,
    NewBag,
    Page,
    PageNumber,
    Part,
    ProgressBar,
    Step,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageClassifier(LabelClassifier):
    """Classifier for building the complete Page element."""

    outputs = frozenset({"page"})
    requires = frozenset(
        {"page_number", "progress_bar", "new_bag", "step", "parts_list"}
    )

    def score(self, result: ClassificationResult) -> None:
        """Create a single page candidate.

        PageClassifier doesn't do any scoring - it just creates a placeholder
        candidate that will be constructed with all page components.
        """
        # Create a simple candidate - all logic is in construct()
        result.add_candidate(
            "page",
            Candidate(
                bbox=result.page_data.bbox,
                label="page",
                score=1.0,
                score_details=None,
                constructed=None,
                source_blocks=[],  # Synthetic element
                failure_reason=None,
            ),
        )

    def construct(self, result: ClassificationResult) -> None:
        """Construct Page elements from candidates."""
        candidates = result.get_candidates("page")
        for candidate in candidates:
            try:
                elem = self._construct_single(candidate, result)
                candidate.constructed = elem
            except Exception as e:
                candidate.failure_reason = str(e)

    def _construct_single(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Construct a Page by collecting all page components.

        Gathers page_number, progress_bar, new_bags, steps, and catalog parts
        to build the complete Page element.
        """
        page_data = result.page_data

        # Get best candidates using score-based selection
        # Extract constructed elements from candidates
        page_number = None
        page_number_candidates = result.get_scored_candidates("page_number")
        if page_number_candidates:
            for pn_candidate in page_number_candidates[:1]:  # Take first
                if pn_candidate.constructed and not pn_candidate.failure_reason:
                    page_number = pn_candidate.constructed
                    assert isinstance(page_number, PageNumber)
                    break

        progress_bar = None
        progress_bar_candidates = result.get_scored_candidates("progress_bar")
        if progress_bar_candidates:
            for pb_candidate in progress_bar_candidates[:1]:  # Take first
                if pb_candidate.constructed and not pb_candidate.failure_reason:
                    progress_bar = pb_candidate.constructed
                    assert isinstance(progress_bar, ProgressBar)
                    break

        # Get new bags from candidates
        new_bags: list[NewBag] = []
        new_bag_candidates = result.get_scored_candidates("new_bag")
        for nb_candidate in new_bag_candidates:
            if nb_candidate.constructed and not nb_candidate.failure_reason:
                new_bag = nb_candidate.constructed
                assert isinstance(new_bag, NewBag)
                new_bags.append(new_bag)

        # Get steps from candidates
        steps: list[Step] = []
        step_candidates = result.get_scored_candidates("step")
        for step_candidate in step_candidates:
            if step_candidate.constructed and not step_candidate.failure_reason:
                step = step_candidate.constructed
                assert isinstance(step, Step)
                steps.append(step)

        # Get all parts from candidates
        all_parts: list[Part] = []
        part_candidates = result.get_scored_candidates("part")
        for part_candidate in part_candidates:
            if part_candidate.constructed and not part_candidate.failure_reason:
                part = part_candidate.constructed
                assert isinstance(part, Part)
                all_parts.append(part)

        # Sort steps by their step_number value
        steps.sort(key=lambda step: step.step_number.value)

        # Collect parts that are already used in steps (to exclude from catalog)
        parts_in_steps: set[int] = set()
        for step in steps:
            for part in step.parts_list.parts:
                parts_in_steps.add(id(part))

        # Filter to get standalone parts (catalog pages) - parts not in steps
        standalone_parts = [p for p in all_parts if id(p) not in parts_in_steps]

        log.debug(
            "[page] Found %d total parts, %d in steps, %d standalone",
            len(all_parts),
            len(parts_in_steps),
            len(standalone_parts),
        )

        # Determine page categories and catalog field
        categories: set[Page.PageType] = set()
        catalog_parts: list[Part] = []

        # Check for instruction content
        if steps:
            categories.add(Page.PageType.INSTRUCTION)

        # Check for catalog content (standalone parts not in steps)
        if standalone_parts:
            categories.add(Page.PageType.CATALOG)
            # Collect standalone parts into catalog
            # Use dict to deduplicate parts by id to avoid having the same
            # Part object appear multiple times
            parts_by_id: dict[int, Part] = {}
            for part in standalone_parts:
                part_id = id(part)
                if part_id in parts_by_id:
                    log.debug(
                        "Skipping duplicate part id:%d in catalog",
                        part_id,
                    )
                parts_by_id[part_id] = part
            catalog_parts = list(parts_by_id.values())
            log.debug(
                "Collected %d unique parts for catalog",
                len(catalog_parts),
            )

        # If no structured content, mark as INFO page
        if not categories:
            categories.add(Page.PageType.INFO)

        log.debug(
            "[page] page=%s categories=%s page_number=%s progress_bar=%s "
            "new_bags=%d steps=%d catalog=%s",
            page_data.page_number,
            [c.name for c in categories],
            page_number.value if page_number else None,
            progress_bar is not None,
            len(new_bags),
            len(steps),
            f"{len(catalog_parts)} parts" if catalog_parts else None,
        )

        return Page(
            bbox=candidate.bbox,
            categories=categories,
            page_number=page_number,
            progress_bar=progress_bar,
            new_bags=new_bags,
            steps=steps,
            catalog=catalog_parts,
            warnings=[],
            unprocessed_elements=[],
        )
