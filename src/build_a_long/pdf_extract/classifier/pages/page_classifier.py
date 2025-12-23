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
    CandidateFailedError,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Background,
    CatalogContent,
    Decoration,
    Divider,
    InfoContent,
    InstructionContent,
    OpenBag,
    Page,
    PageNumber,
    Part,
    Preview,
    ProgressBar,
    Scale,
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
            "decoration",
            "divider",
            "page_number",
            "preview",
            "progress_bar",
            "open_bag",
            "scale",
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

    def _build_page_number(self, result: ClassificationResult) -> PageNumber | None:
        """Build the page number element if available."""
        candidates = result.get_scored_candidates("page_number")
        if candidates:
            try:
                page_number = result.build(candidates[0])
                assert isinstance(page_number, PageNumber)
                return page_number
            except CandidateFailedError as e:
                log.debug("Failed to build page_number: %s", e)
        return None

    def _build_progress_bar(self, result: ClassificationResult) -> ProgressBar | None:
        """Build the progress bar element if available.

        The solver constrains progress_bar to at most one, so we build all
        selected candidates and return the first (if any).
        """
        progress_bars = result.build_all_for_label("progress_bar")
        if progress_bars:
            assert len(progress_bars) == 1
            progress_bar = progress_bars[0]
            assert isinstance(progress_bar, ProgressBar)
            return progress_bar
        return None

    def _build_background(self, result: ClassificationResult) -> Background | None:
        """Build the background element if available."""
        candidates = result.get_scored_candidates("background")
        if candidates:
            try:
                background = result.build(candidates[0])
                assert isinstance(background, Background)
                return background
            except CandidateFailedError as e:
                log.debug("Failed to build background: %s", e)
        return None

    def _build_trivia_text(self, result: ClassificationResult) -> TriviaText | None:
        """Build the trivia text element if available."""
        candidates = result.get_scored_candidates("trivia_text")
        if candidates:
            try:
                trivia_text = result.build(candidates[0])
                assert isinstance(trivia_text, TriviaText)
                return trivia_text
            except CandidateFailedError as e:
                log.debug("Failed to build trivia_text: %s", e)
        return None

    def _build_scale(self, result: ClassificationResult) -> Scale | None:
        """Build the scale indicator if available."""
        candidates = result.get_scored_candidates("scale")
        if candidates:
            try:
                scale = result.build(candidates[0])
                assert isinstance(scale, Scale)
                return scale
            except CandidateFailedError as e:
                log.debug("Failed to build scale: %s", e)
        return None

    def _build_dividers(self, result: ClassificationResult) -> list[Divider]:
        """Build all divider elements."""
        dividers = result.build_all_for_label("divider")
        assert all(isinstance(d, Divider) for d in dividers)
        return [d for d in dividers if isinstance(d, Divider)]

    def _build_decorations(self, result: ClassificationResult) -> list[Decoration]:
        """Build all decoration elements (INFO page content)."""
        decorations = result.build_all_for_label("decoration")
        assert all(isinstance(d, Decoration) for d in decorations)
        return [d for d in decorations if isinstance(d, Decoration)]

    def _build_open_bags(self, result: ClassificationResult) -> list[OpenBag]:
        """Build all open bag elements."""
        open_bags: list[OpenBag] = []
        for candidate in result.get_scored_candidates("open_bag"):
            try:
                elem = result.build(candidate)
                assert isinstance(elem, OpenBag)
                open_bags.append(elem)
            except Exception as e:
                log.debug(
                    "Failed to construct open_bag candidate at %s: %s",
                    candidate.bbox,
                    e,
                )
        return open_bags

    def _build_parts(self, result: ClassificationResult) -> list[Part]:
        """Build all part elements."""
        parts: list[Part] = []
        for candidate in result.get_scored_candidates("part"):
            try:
                elem = result.build(candidate)
                assert isinstance(elem, Part)
                parts.append(elem)
            except Exception as e:
                log.debug(
                    "Failed to construct part candidate at %s: %s",
                    candidate.bbox,
                    e,
                )
        return parts

    def _build_steps(self, result: ClassificationResult) -> list[Step]:
        """Build all step elements.

        This uses StepClassifier's coordinated build_all which handles:
        1. Building rotation symbols first (so they consume Drawing blocks)
        2. Building all Step candidates
        3. Hungarian matching to assign rotation symbols to steps
        """
        steps = result.build_all_for_label("step")
        assert all(isinstance(s, Step) for s in steps)
        steps = [s for s in steps if isinstance(s, Step)]
        steps.sort(key=lambda step: step.step_number.value)
        return steps

    def _collect_previews(self, result: ClassificationResult) -> list[Preview]:
        """Collect already-built preview elements.

        Previews are built by StepClassifier.build_all alongside subassemblies
        to properly deconflict white boxes that could be either.
        """
        previews = []
        for c in result.get_candidates("preview"):
            constructed = result.get_constructed(c)
            if constructed is not None and isinstance(constructed, Preview):
                previews.append(constructed)
        return previews

    def _determine_categories(
        self,
        steps: list[Step],
        standalone_parts: list[Part],
        open_bags: list[OpenBag],
    ) -> set[Page.PageType]:
        """Determine page categories based on content."""
        categories: set[Page.PageType] = set()

        # TODO Should this be a property on Page, that is computed on access?
        if steps or open_bags:
            categories.add(Page.PageType.INSTRUCTION)

        if standalone_parts:
            categories.add(Page.PageType.CATALOG)

        if not categories:
            categories.add(Page.PageType.INFO)

        return categories

    def _get_standalone_parts(
        self, all_parts: list[Part], steps: list[Step]
    ) -> list[Part]:
        """Filter parts to get standalone catalog parts (not in steps).

        Uses Part.id for identity comparison since each Part gets a unique
        auto-generated ID at construction time.
        """
        # Collect IDs of parts that appear in steps
        parts_in_steps: set[int] = set()
        for step in steps:
            if step.parts_list:
                for part in step.parts_list.parts:
                    parts_in_steps.add(part.id)

        # Filter to parts whose ID doesn't appear in any step
        standalone = [p for p in all_parts if p.id not in parts_in_steps]

        log.debug(
            "[page] Found %d total parts, %d in steps, %d standalone",
            len(all_parts),
            len(parts_in_steps),
            len(standalone),
        )

        return standalone

    def build(self, candidate: Candidate, result: ClassificationResult) -> Page:
        """Construct a Page by collecting all page components.

        Gathers page_number, progress_bar, open_bags, steps, and catalog parts
        to build the complete Page element.
        """
        page_data = result.page_data

        # Build individual elements
        page_number = self._build_page_number(result)
        progress_bar = self._build_progress_bar(result)
        background = self._build_background(result)
        trivia_text = self._build_trivia_text(result)
        scale = self._build_scale(result)

        # Build collections
        dividers = self._build_dividers(result)
        decorations = self._build_decorations(result)
        open_bags = self._build_open_bags(result)
        all_parts = self._build_parts(result)
        steps = self._build_steps(result)
        previews = self._collect_previews(result)

        # Determine catalog parts and page categories
        catalog_parts = self._get_standalone_parts(all_parts, steps)
        categories = self._determine_categories(steps, catalog_parts, open_bags)

        # Build composed content objects based on categories
        instruction_content: InstructionContent | None = None
        if Page.PageType.INSTRUCTION in categories:
            instruction_content = InstructionContent(
                steps=steps,
                open_bags=open_bags,
            )

        catalog_content: CatalogContent | None = None
        if Page.PageType.CATALOG in categories:
            catalog_content = CatalogContent(parts=catalog_parts)

        info_content: InfoContent | None = None
        if Page.PageType.INFO in categories:
            info_content = InfoContent(decorations=decorations)

        log.debug(
            "[page] page=%s categories=%s page_number=%s progress_bar=%s "
            "background=%s dividers=%d decorations=%d previews=%d "
            "open_bags=%d steps=%d catalog=%s",
            page_data.page_number,
            [c.name for c in categories],
            page_number.value if page_number else None,
            progress_bar is not None,
            background is not None,
            len(dividers),
            len(decorations),
            len(previews),
            len(open_bags),
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
            scale=scale,
            previews=previews,
            trivia_text=trivia_text,
            unconsumed_blocks_count=result.count_unconsumed_blocks(),
            instruction=instruction_content,
            catalog=catalog_content,
            info=info_content,
        )
