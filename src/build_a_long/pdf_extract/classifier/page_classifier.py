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

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create a Page candidate.

        Collects page_number, progress_bar, and step elements to build a
        complete Page. Uses get_winners_by_score() to select the best
        candidates based on scores.

        For catalog pages (pages with parts_list but no steps), creates a
        catalog Page with all parts_lists merged into a single catalog field.
        """
        page_data = result.page_data

        # Get best candidates using score-based selection
        # (max_count=1 for singleton elements)
        page_number_winners = result.get_winners_by_score(
            "page_number", PageNumber, max_count=1
        )
        page_number = page_number_winners[0] if page_number_winners else None

        progress_bar_winners = result.get_winners_by_score(
            "progress_bar", ProgressBar, max_count=1
        )
        progress_bar = progress_bar_winners[0] if progress_bar_winners else None

        # Get new bags using score-based selection
        new_bags = result.get_winners_by_score("new_bag", NewBag)

        # Get steps using score-based selection (StepClassifier now handles
        # deduplication in evaluate(), so all step candidates are valid)
        steps = result.get_winners_by_score("step", Step)

        # Get standalone parts (catalog pages) - these won't be in parts_lists
        standalone_parts = result.get_winners_by_score("part", Part)

        # Sort steps by their step_number value
        steps.sort(key=lambda step: step.step_number.value)

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

        # Construct the Page
        constructed = Page(
            bbox=page_data.bbox,
            categories=categories,
            page_number=page_number,
            progress_bar=progress_bar,
            new_bags=new_bags,
            steps=steps,
            catalog=catalog_parts,
            warnings=[],
            unprocessed_elements=[],
        )

        # Add candidate
        result.add_candidate(
            "page",
            Candidate(
                bbox=page_data.bbox,
                label="page",
                score=1.0,
                score_details=None,
                constructed=constructed,
                source_block=None,  # Synthetic element
                failure_reason=None,
            ),
        )
