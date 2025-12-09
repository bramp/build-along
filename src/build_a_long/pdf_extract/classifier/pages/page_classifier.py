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
    Background,
    Divider,
    NewBag,
    Page,
    PageNumber,
    Part,
    ProgressBar,
    Step,
    TriviaText,
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


class PageClassifier(LabelClassifier):
    """Classifier for building the complete Page element."""

    output = "page"
    requires = frozenset(
        {
            "background",
            "divider",
            "page_number",
            # TODO: Re-enable preview once classifier is more accurate
            # "preview",
            "progress_bar",
            "new_bag",
            "step",
            "parts_list",
            "rotation_symbol",
            "trivia_text",
        }
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

        # Build the background (if any) - only one per page
        background = None
        background_candidates = result.get_scored_candidates(
            "background", valid_only=False, exclude_failed=True
        )
        if background_candidates:
            best_cand = background_candidates[0]
            background = result.build(best_cand)
            assert isinstance(background, Background)

        # Build all dividers
        dividers = result.build_all_for_label("divider")
        assert all(isinstance(d, Divider) for d in dividers)
        dividers = [d for d in dividers if isinstance(d, Divider)]  # type narrow

        # TODO: Re-enable preview building once classifier is more accurate
        # Build all previews
        # previews = result.build_all_for_label("preview")
        # assert all(isinstance(p, Preview) for p in previews)
        # previews = [p for p in previews if isinstance(p, Preview)]  # type narrow
        previews = []  # Disabled for now

        # Build trivia text (if any) - only one per page
        # TODO Consider multiple per page
        trivia_text = None
        trivia_text_candidates = result.get_scored_candidates(
            "trivia_text", valid_only=False, exclude_failed=True
        )
        if trivia_text_candidates:
            best_cand = trivia_text_candidates[0]
            trivia_text = result.build(best_cand)
            assert isinstance(trivia_text, TriviaText)

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
                log.debug(
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
                log.debug(
                    "Failed to construct part candidate at %s: %s",
                    part_candidate.bbox,
                    e,
                )

        # Build all steps using the StepClassifier's coordinated build_all.
        # This handles:
        # 1. Building rotation symbols first (so they claim Drawing blocks)
        # 2. Building all Step candidates
        # 3. Hungarian matching to assign rotation symbols to steps
        steps = result.build_all_for_label("step")
        assert all(isinstance(s, Step) for s in steps)
        steps = [s for s in steps if isinstance(s, Step)]  # type narrow

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
            "background=%s dividers=%d previews=%d new_bags=%d steps=%d catalog=%s",
            page_data.page_number,
            [c.name for c in categories],
            page_number.value if page_number else None,
            progress_bar is not None,
            background is not None,
            len(dividers),
            len(previews),
            len(new_bags),
            len(steps),
            f"{len(catalog_parts)} parts" if catalog_parts else None,
        )

        return Page(
            bbox=candidate.bbox,
            pdf_page_number=page_data.page_number,
            categories=categories,
            page_number=page_number,
            progress_bar=progress_bar,
            background=background,
            dividers=dividers,
            previews=previews,
            trivia_text=trivia_text,
            new_bags=new_bags,
            steps=steps,
            catalog=catalog_parts,
        )
