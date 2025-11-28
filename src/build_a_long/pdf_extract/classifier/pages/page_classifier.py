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

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    NewBag,
    Page,
    PageNumber,
    Part,
    ProgressBar,
    Step,
)

log = logging.getLogger(__name__)


class _PageScore(Score, BaseModel):
    """Score details for Page candidates.

    PageClassifier always succeeds with score 1.0 since it's a synthetic
    element that aggregates other classified components.
    """

    def score(self) -> float:
        """Return the score value (always 1.0 for pages)."""
        return 1.0


@dataclass(frozen=True)
class PageClassifier(LabelClassifier):
    """Classifier for building the complete Page element."""

    output = "page"
    requires = frozenset(
        {"page_number", "progress_bar", "new_bag", "step", "parts_list"}
    )

    def _score(self, result: ClassificationResult) -> None:
        """Create a single page candidate.

        PageClassifier doesn't do complex scoring - it just creates a candidate
        that will be constructed with all page components.
        """
        # Create a candidate with score_details for consistency
        result.add_candidate(
            Candidate(
                bbox=result.page_data.bbox,
                label="page",
                score=1.0,
                score_details=_PageScore(),
                source_blocks=[],  # Synthetic element
            ),
        )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Page:
        """Construct a Page by collecting all page components.

        Gathers page_number, progress_bar, new_bags, steps, and catalog parts
        to build the complete Page element.
        """
        page_data = result.page_data

        # Get best candidates using score-based selection
        # get_scored_candidates returns only valid candidates by default, so
        # we must set valid_only=False, exclude_failed=True to get
        # candidates that haven't been constructed yet.
        page_number = None
        page_number_candidates = result.get_scored_candidates(
            "page_number", valid_only=False, exclude_failed=True
        )
        if page_number_candidates:
            best_cand = page_number_candidates[0]
            page_number = result.build(best_cand)
            assert isinstance(page_number, PageNumber)

        progress_bar = None
        progress_bar_candidates = result.get_scored_candidates(
            "progress_bar", valid_only=False, exclude_failed=True
        )
        if progress_bar_candidates:
            best_cand = progress_bar_candidates[0]
            progress_bar = result.build(best_cand)
            assert isinstance(progress_bar, ProgressBar)

        # Get new bags from candidates
        new_bags: list[NewBag] = []

        # Construct ALL new_bag candidates? Or just the ones that pass a threshold?
        # For now, construct all scored candidates.
        # TODO Consider pre-filtering based on runs of bag numbers
        for nb_candidate in result.get_scored_candidates(
            "new_bag", valid_only=False, exclude_failed=True
        ):
            try:
                elem = result.build(nb_candidate)
                assert isinstance(elem, NewBag)
                new_bags.append(elem)
            except Exception as e:
                log.warning(
                    "Failed to construct new_bag candidate at %s: %s",
                    nb_candidate.bbox,
                    e,
                )

        # Get all parts from candidates first, as they typically have more
        # useful context.
        all_parts: list[Part] = []
        for part_candidate in result.get_scored_candidates(
            "part", valid_only=False, exclude_failed=True
        ):
            try:
                elem = result.build(part_candidate)
                assert isinstance(elem, Part)
                all_parts.append(elem)
            except Exception as e:
                log.warning(
                    "Failed to construct part candidate at %s: %s",
                    part_candidate.bbox,
                    e,
                )

        # Get steps from candidates
        # TODO Consider pre-filtering based on runs of step numbers
        steps: list[Step] = []
        for step_candidate in result.get_scored_candidates(
            "step", valid_only=False, exclude_failed=True
        ):
            try:
                elem = result.build(step_candidate)
                assert isinstance(elem, Step)
                steps.append(elem)
            except Exception as e:
                log.warning(
                    "Failed to construct step candidate at %s: %s",
                    step_candidate.bbox,
                    e,
                )

        # Sort steps by their step_number value
        steps.sort(key=lambda step: step.step_number.value)

        # Collect parts that are already used in steps (to exclude from catalog)
        parts_in_steps: set[int] = set()
        for step in steps:
            if step.parts_list:
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
        catalog_parts: list[Part] = standalone_parts

        # Check for instruction content
        if steps:
            categories.add(Page.PageType.INSTRUCTION)

        # Check for catalog content (standalone parts not in steps)
        if standalone_parts:
            categories.add(Page.PageType.CATALOG)

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
        )
