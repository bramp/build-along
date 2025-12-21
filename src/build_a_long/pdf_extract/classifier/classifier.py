"""
Rule-based classifier for labeling page elements.

Pipeline order and dependencies
--------------------------------
The classification pipeline operates in two main phases:

1. **Bottom-up Scoring**: All classifiers run independently to identify potential
   candidates (e.g. page numbers, part counts, step numbers) and score them based
   on heuristics. No construction of final elements happens here.

2. **Top-down Construction**: The root `PageClassifier` is invoked to construct
   the final `Page` object. It recursively requests the construction of its
   dependencies (e.g. "Give me the best PageNumber"), which in turn construct
   their own dependencies. This ensures a consistent and validated object tree.

"""

from __future__ import annotations

import logging

from build_a_long.pdf_extract.classifier.bags import (
    BagNumberClassifier,
    LoosePartSymbolClassifier,
    OpenBagClassifier,
)
from build_a_long.pdf_extract.classifier.batch_classification_result import (
    BatchClassificationResult,
)
from build_a_long.pdf_extract.classifier.block_filter import (
    filter_duplicate_blocks,
    filter_overlapping_text_blocks,
)
from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.pages import (
    PageHintCollection,
)
from build_a_long.pdf_extract.classifier.pages.background_classifier import (
    BackgroundClassifier,
)
from build_a_long.pdf_extract.classifier.pages.divider_classifier import (
    DividerClassifier,
)
from build_a_long.pdf_extract.classifier.pages.full_page_background_classifier import (
    FullPageBackgroundClassifier,
)
from build_a_long.pdf_extract.classifier.pages.info_page_decoration_classifier import (
    InfoPageDecorationClassifier,
)
from build_a_long.pdf_extract.classifier.pages.page_classifier import PageClassifier
from build_a_long.pdf_extract.classifier.pages.page_edge_classifier import (
    PageEdgeClassifier,
)
from build_a_long.pdf_extract.classifier.pages.page_number_classifier import (
    PageNumberClassifier,
)
from build_a_long.pdf_extract.classifier.pages.preview_classifier import (
    PreviewClassifier,
)
from build_a_long.pdf_extract.classifier.pages.progress_bar_bar_classifier import (
    ProgressBarBarClassifier,
)
from build_a_long.pdf_extract.classifier.pages.progress_bar_classifier import (
    ProgressBarClassifier,
)
from build_a_long.pdf_extract.classifier.pages.progress_bar_indicator_classifier import (  # noqa: E501
    ProgressBarIndicatorClassifier,
)
from build_a_long.pdf_extract.classifier.pages.trivia_text_classifier import (
    TriviaTextClassifier,
)
from build_a_long.pdf_extract.classifier.parts import (
    PartCountClassifier,
    PartNumberClassifier,
    PartsClassifier,
    PartsImageClassifier,
    PartsListClassifier,
    PieceLengthClassifier,
    ScaleClassifier,
    ScaleTextClassifier,
    ShineClassifier,
)
from build_a_long.pdf_extract.classifier.removal_reason import RemovalReason
from build_a_long.pdf_extract.classifier.steps import (
    ArrowClassifier,
    DiagramClassifier,
    RotationSymbolClassifier,
    StepClassifier,
    StepCountClassifier,
    StepNumberClassifier,
    SubAssemblyClassifier,
    SubStepClassifier,
    SubStepNumberClassifier,
)
from build_a_long.pdf_extract.classifier.text import FontSizeHints, TextHistogram
from build_a_long.pdf_extract.classifier.topological_sort import topological_sort
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Blocks

logger = logging.getLogger(__name__)

# Pages with more blocks than this threshold will be skipped during classification.
# This avoids O(n²) algorithms (like duplicate detection) that become prohibitively
# slow on pages with thousands of vector drawings. Such pages are typically info
# pages where each character is a separate vector graphic.
# TODO: Add spatial indexing to handle high-block pages efficiently.
MAX_BLOCKS_PER_PAGE = 1000


# TODO require config, so we don't accidentally use default empty config
def classify_elements(
    page: PageData, config: ClassifierConfig | None = None
) -> ClassificationResult:
    """Classify and label elements on a single page using rule-based heuristics.

    Args:
        page: A single PageData object to classify.
        config: Optional classifier configuration with font/page hints.
            If None, uses default empty configuration (no hints).
            For better classification accuracy, pass a config with
            FontSizeHints computed from multiple pages of the same PDF.

    Returns:
        A ClassificationResult object containing the classification results.
    """
    if config is None:
        config = ClassifierConfig()
    classifier = Classifier(config)

    return classifier.classify(page)


def classify_pages(
    pages: list[PageData], pages_for_hints: list[PageData] | None = None
) -> BatchClassificationResult:
    """Classify and label elements across multiple pages using rule-based heuristics.

    This function performs a three-phase process:
    1. Filtering phase: Mark duplicate/similar blocks as removed on each page
    2. Analysis phase: Build font size hints from text properties (excluding
       removed blocks)
    3. Classification phase: Use hints to guide element classification

    Args:
        pages: A list of PageData objects to classify.
        pages_for_hints: Optional list of pages to use for generating font/page hints.
            If None, uses `pages`. This allows generating hints from all pages
            while only classifying a subset (e.g., when using --pages filter).

    Returns:
        BatchClassificationResult containing per-page results and global histogram
    """

    # TODO There is a bunch of duplication in here between hints and non-hints. Refactor

    # Use all pages for hint generation if provided, otherwise use selected pages
    hint_pages = pages_for_hints if pages_for_hints is not None else pages

    # Phase 1: Filter duplicate blocks on each page and track removals
    # Skip pages with too many blocks to avoid O(n²) performance issues
    removed_blocks_per_page: list[dict[Blocks, RemovalReason]] = []
    skipped_pages: set[int] = set()  # Track page numbers that are skipped

    for page_data in pages:
        # Skip pages with too many blocks - these are likely info/inventory pages
        # with vectorized text that cause O(n²) algorithms to be very slow
        if len(page_data.blocks) > MAX_BLOCKS_PER_PAGE:
            logger.debug(
                f"Page {page_data.page_number}: skipping classification "
                f"({len(page_data.blocks)} blocks exceeds threshold of "
                f"{MAX_BLOCKS_PER_PAGE})"
            )
            skipped_pages.add(page_data.page_number)
            removed_blocks_per_page.append({})
            continue

        kept_blocks = page_data.blocks

        # Filter overlapping text blocks (e.g., "4" and "43" at same origin)
        kept_blocks, text_removed = filter_overlapping_text_blocks(kept_blocks)

        # Filter duplicate image/drawing blocks based on IOU
        kept_blocks, bbox_removed = filter_duplicate_blocks(kept_blocks)

        # Combine all removal mappings into a single dict for this page
        combined_removed_mapping = {
            **text_removed,
            **bbox_removed,
        }

        logger.debug(
            f"Page {page_data.page_number}: "
            f"filtered {len(text_removed)} overlapping text, "
            f"{len(bbox_removed)} duplicate bbox blocks"
        )

        removed_blocks_per_page.append(combined_removed_mapping)

    # Phase 2: Extract font size hints from hint pages (excluding removed blocks)
    # Build pages with non-removed blocks for hint extraction and histogram

    # Filter duplicates from hint pages (may be different from pages to classify)
    hint_pages_without_duplicates = []
    for page_data in hint_pages:
        # Skip high-block pages for hints too (same threshold)
        if len(page_data.blocks) > MAX_BLOCKS_PER_PAGE:
            continue

        # TODO We are re-filtering duplicates here; optimize by changing the API
        # to accept one list of PageData, and seperate by page_numbers.
        kept_blocks = page_data.blocks
        kept_blocks, _ = filter_overlapping_text_blocks(kept_blocks)
        kept_blocks, _ = filter_duplicate_blocks(kept_blocks)

        hint_pages_without_duplicates.append(
            PageData(
                page_number=page_data.page_number,
                bbox=page_data.bbox,
                blocks=kept_blocks,
            )
        )

    # Build pages without duplicates for classification
    pages_without_duplicates = []
    for page_data, removed_mapping in zip(pages, removed_blocks_per_page, strict=True):
        # We need to filter blocks that were removed by ANY filter
        non_removed_blocks = [
            block for block in page_data.blocks if block not in removed_mapping
        ]
        pages_without_duplicates.append(
            PageData(
                page_number=page_data.page_number,
                bbox=page_data.bbox,
                blocks=non_removed_blocks,
            )
        )

    # Generate hints from hint pages, histogram from pages to classify
    font_size_hints = FontSizeHints.from_pages(hint_pages_without_duplicates)
    page_hints = PageHintCollection.from_pages(hint_pages_without_duplicates)
    histogram = TextHistogram.from_pages(pages_without_duplicates)

    # Phase 3: Classify using the hints (on pages without duplicates)
    config = ClassifierConfig(font_size_hints=font_size_hints, page_hints=page_hints)
    classifier = Classifier(config)

    results = []
    for page_data, page_without_duplicates, removed_mapping in zip(
        pages, pages_without_duplicates, removed_blocks_per_page, strict=True
    ):
        # Handle skipped pages
        if page_data.page_number in skipped_pages:
            result = ClassificationResult(
                page_data=page_data,
                skipped_reason=(
                    f"Page has {len(page_data.blocks)} blocks, which exceeds "
                    f"the threshold of {MAX_BLOCKS_PER_PAGE}. This is likely an "
                    f"info/inventory page with vectorized text."
                ),
            )
            results.append(result)
            continue

        # Classify using only non-removed blocks
        result = classifier.classify(page_without_duplicates)

        # Update result to use original page_data (with all blocks)
        result.page_data = page_data

        # Mark removed blocks
        for removed_block, removal_reason in removed_mapping.items():
            result.mark_removed(removed_block, removal_reason)

        results.append(result)

    return BatchClassificationResult(results=results, histogram=histogram)


type Classifiers = (
    PageNumberClassifier
    | ProgressBarBarClassifier
    | ProgressBarClassifier
    | ProgressBarIndicatorClassifier
    | PreviewClassifier
    | FullPageBackgroundClassifier
    | PageEdgeClassifier
    | BackgroundClassifier
    | DividerClassifier
    | InfoPageDecorationClassifier
    | BagNumberClassifier
    | PartCountClassifier
    | PartNumberClassifier
    | StepNumberClassifier
    | StepCountClassifier
    | PieceLengthClassifier
    | ScaleClassifier
    | ScaleTextClassifier
    | PartsClassifier
    | PartsListClassifier
    | PartsImageClassifier
    | ShineClassifier
    | OpenBagClassifier
    | LoosePartSymbolClassifier
    | DiagramClassifier
    | ArrowClassifier
    | SubAssemblyClassifier
    | StepClassifier
    | TriviaTextClassifier
    | PageClassifier
)


class Classifier:
    """Performs a single run of classification based on rules, configuration, and hints.

    This class orchestrates the two-phase classification process:
    1. **Scoring Phase**: All classifiers run `_score()` to create candidates
    2. **Construction Phase**: PageClassifier.build_all() triggers top-down construction

    This class should be stateless.

    Best Practices for Writing Classifiers
    =======================================

    Phase 1: Scoring (`_score()` method)
    ------------------------------------

    The scoring phase evaluates blocks and creates candidates. Key rules:

    **Allowed API Access:**
    - `result.page_data.blocks` - Access all page blocks
    - `result.get_candidates(label)` - Get candidates for a label
    - `result.get_scored_candidates(label)` - Get scored candidates (identical to
      get_candidates() during scoring phase since nothing is built yet)
    - IMPORTANT: Only request candidates for labels in your `requires` frozenset

    **Scoring Philosophy:**
    - Score based on INTRINSIC properties (size, position, text content, color)
    - Observe potential relationships to inform score ("could have 3 children")
    - DO NOT pre-assign specific child candidates in your scoring logic
    - DO NOT check `result.is_consumed()` - that's for the build phase

    **Score Object Requirements:**
    - MUST inherit from the `Score` abstract base class
    - SHOULD store candidate references from dependencies (e.g.,
      `part_count_candidate: Candidate`)
    - Should NOT store Block objects directly
    - Reason: Makes it clear if scoring depends on a built candidate or not

    **Example:**

    .. code-block:: python

        class MyScore(Score):
            # Score with dependency candidate reference.
            intrinsic_score: float
            child_candidate: Candidate | None  # OK: Store candidate
            # child_block: Block | None  # BAD: Don't store blocks

            def score(self) -> Weight:
                return self.intrinsic_score

        def _score(self, result: ClassificationResult) -> None:
            # Get dependency candidates (only if in self.requires)
            child_candidates = result.get_scored_candidates("child_label")

            for block in result.page_data.blocks:
                # Score based on intrinsic properties
                intrinsic_score = self._calculate_intrinsic_score(block)

                # Optional: observe potential children to inform score
                best_child = None
                if child_candidates:
                    best_child = self._find_closest(block, child_candidates)
                    if best_child:
                        intrinsic_score += 0.2  # Boost if potential child exists

                score_obj = MyScore(
                    intrinsic_score=intrinsic_score,
                    child_candidate=best_child,  # Store candidate reference
                )

                result.add_candidate(
                    Candidate(
                        bbox=block.bbox,
                        label=self.output,
                        score=score_obj.score(),
                        score_details=score_obj,
                        source_blocks=[block],
                    )
                )

    Phase 2: Construction (`build()` method)
    ----------------------------------------

    The build phase constructs LegoPageElements from winning candidates. Key rules:

    **Construction Process:**
    - Validate that dependency candidates are still valid (not consumed/failed)
    - Use `result.build(candidate)` to construct child elements
    - Discover relationships at build time (don't rely on pre-scored relationships)
    - Check `result.is_consumed()` if searching for available blocks

    **Source Blocks Rules:**
    - A source block should only be assigned to ONE built candidate
    - Multiple candidates can reference a block during scoring, but only one builds
    - **Non-composite elements**: MUST have 1+ source blocks
    - **Composite elements**: MAY have 0+ source blocks (decoration, borders)
    - Parent's source_blocks should NOT include child's source_blocks

    **Exception Handling:**
    - Raise `CandidateFailedError` for intentional build failures
    - Let other exceptions (TypeError, AttributeError) propagate naturally
    - Caller should catch exceptions if element is optional or alternatives exist
    - Otherwise, let exceptions bubble up

    **Example:**

    .. code-block:: python

        def build(
            self, candidate: Candidate, result: ClassificationResult
        ) -> MyElement:
            # Construct element from candidate.
            score = candidate.score_details
            assert isinstance(score, MyScore)

            # Build child if candidate is still valid
            child_elem = None
            if score.child_candidate:
                try:
                    child_elem = result.build(score.child_candidate)
                    assert isinstance(child_elem, ChildElement)
                except CandidateFailedError:
                    # Child failed - either fail or continue without it
                    if self._requires_child:
                        raise  # Propagate failure
                    # Otherwise continue with child_elem = None

            return MyElement(
                bbox=candidate.bbox,
                child=child_elem,
                # source_blocks inherited from candidate
            )

    Phase 2b: Global Coordination (`build_all()` method)
    ----------------------------------------------------

    Most classifiers use the default `build_all()` which iterates through
    candidates and calls `build()` on each. Override when you need:

    **When to Override:**
    - Global optimization (e.g., Hungarian matching to find N best pairings)
    - Building multiple candidates with interdependencies
    - Pre-build setup that affects all candidates

    **Key Differences:**
    - `build()`: Works in isolation on a single candidate
    - `build_all()`: Coordinates multiple candidates globally

    **Can build_all() call other labels' builds?**
    - Technically yes, but best to avoid unless necessary
    - Usually each classifier manages only its own label's candidates

    **Example:**

    .. code-block:: python

        def build_all(self, result: ClassificationResult) -> list[LegoPageElements]:
            # Build candidates using Hungarian matching.
            candidates = result.get_candidates(self.output)

            # Perform global optimization
            best_assignments = self._hungarian_match(candidates)

            elements = []
            for candidate in best_assignments:
                try:
                    elem = result.build(candidate)
                    elements.append(elem)
                except CandidateFailedError as e:
                    log.debug(f\"Failed to build {candidate.label}: {e}\")
                    log.debug(f"Failed to build {candidate.label}: {e}")

            return elements

    Common Patterns
    ---------------

    **Pattern 1: Atomic Classifier (single block → element)**

    **Recommendation**: Use `RuleBasedClassifier` for most atomic classifiers.
    It provides a declarative, maintainable way to score blocks using composable
    rules. Only implement custom `_score()` logic when you need complex pairing
    or non-standard scoring that can't be expressed with rules.

    .. code-block:: python

        class MyAtomicClassifier(RuleBasedClassifier):
            output = "my_label"
            requires = frozenset()  # No dependencies

            @property
            def rules(self) -> Sequence[Rule]:
                return [
                    IsInstanceFilter((Text,)),
                    PositionScore(...),
                    # ... more rules
                ]

            def build(self, candidate, result) -> MyElement:
                return MyElement(bbox=candidate.bbox)

    **Pattern 2: Composite Classifier (combines other elements)**

    .. code-block:: python

        class MyCompositeClassifier(LabelClassifier):
            output = "my_composite"
            requires = frozenset({"child1", "child2"})

            def _score(self, result):
                child1_cands = result.get_scored_candidates("child1")
                child2_cands = result.get_scored_candidates("child2")

                # Create composite candidates by pairing children
                for c1 in child1_cands:
                    for c2 in child2_cands:
                        if self._are_related(c1, c2):
                            score = self._compute_pair_score(c1, c2)
                            result.add_candidate(
                                Candidate(
                                    bbox=BBox.union(c1.bbox, c2.bbox),
                                    label=self.output,
                                    score=score.score(),
                                    score_details=score,
                                    source_blocks=[],  # Composite
                                )
                            )

            def build(self, candidate, result) -> MyComposite:
                score = candidate.score_details
                child1 = result.build(score.child1_candidate)
                child2 = result.build(score.child2_candidate)
                return MyComposite(bbox=candidate.bbox, c1=child1, c2=child2)

    See Also
    --------
    - classifier/DESIGN.md: Architectural principles
    - classifier/README.md: Classification pipeline overview
    - LabelClassifier: Base class for all classifiers
    - RuleBasedClassifier: Rule-based classifier base class
    """

    def __init__(self, config: ClassifierConfig, use_constraint_solver: bool = False):
        """Initialize the classifier with optional constraint solver.

        Args:
            config: Classifier configuration with hints and settings
            use_constraint_solver: If True, use CP-SAT solver to select candidates
                before construction. If False, use traditional greedy/speculative
                building approach.
        """
        self.config = config
        self.use_constraint_solver = use_constraint_solver

        # Sort classifiers topologically based on their dependencies
        self.classifiers = topological_sort(
            [
                PageNumberClassifier(config=config),
                ProgressBarIndicatorClassifier(config=config),
                ProgressBarBarClassifier(config=config),
                ProgressBarClassifier(config=config),
                FullPageBackgroundClassifier(config=config),
                PageEdgeClassifier(config=config),
                BackgroundClassifier(config=config),
                DividerClassifier(config=config),
                InfoPageDecorationClassifier(config=config),
                BagNumberClassifier(config=config),
                PartCountClassifier(config=config),
                PartNumberClassifier(config=config),
                StepNumberClassifier(config=config),
                SubStepNumberClassifier(config=config),
                StepCountClassifier(config=config),
                PieceLengthClassifier(config=config),
                ScaleTextClassifier(config=config),
                ScaleClassifier(config=config),
                PartsClassifier(config=config),
                PartsListClassifier(config=config),
                DiagramClassifier(config=config),
                RotationSymbolClassifier(config=config),
                ArrowClassifier(config=config),
                PartsImageClassifier(config=config),
                ShineClassifier(config=config),
                OpenBagClassifier(config=config),
                LoosePartSymbolClassifier(config=config),
                PreviewClassifier(config=config),
                SubStepClassifier(config=config),
                SubAssemblyClassifier(config=config),
                StepClassifier(config=config),
                TriviaTextClassifier(config=config),
                PageClassifier(config=config),
            ]
        )

    def classify(self, page_data: PageData) -> ClassificationResult:
        """
        Runs the classification logic and returns a result.
        It does NOT modify page_data directly.

        The classification process runs in three phases:
        1. Score all classifiers (bottom-up) - auto-registers classifiers
        2. [Optional] Run constraint solver to select candidates
        3. Construct final elements (top-down starting from Page)
        """
        result = ClassificationResult(page_data=page_data)

        logger.debug(f"Starting classification for page {page_data.page_number}")

        # 1. Score all classifiers (Bottom-Up)
        # Note: score() automatically registers each classifier for its output labels
        for classifier in self.classifiers:
            classifier.score(result)

        # 2. [Optional] Run constraint solver to select candidates
        if self.use_constraint_solver:
            self._run_constraint_solver(result)

        # 3. Construct (Top-Down)
        # Find the PageClassifier to start the construction process
        page_classifier = next(
            c for c in self.classifiers if isinstance(c, PageClassifier)
        )
        page_classifier.build_all(result)

        # 4. Validate classification invariants
        self._validate_classification_result(result)

        return result

    def _run_constraint_solver(self, result: ClassificationResult) -> None:
        """Run CP-SAT constraint solver to select candidates.

        This method:
        1. Creates a ConstraintModel and adds all candidates
        2. Calls declare_constraints() on each classifier
        3. Runs auto-generation of schema-based constraints
        4. Solves the constraint problem
        5. Marks selected candidates in the result

        Args:
            result: The classification result with scored candidates
        """
        from build_a_long.pdf_extract.classifier.constraint_model import (  # noqa: PLC0415
            ConstraintModel,
        )
        from build_a_long.pdf_extract.classifier.schema_constraint_generator import (  # noqa: PLC0415
            SchemaConstraintGenerator,
        )

        logger.debug(
            f"Running constraint solver for page {result.page_data.page_number}"
        )

        # Create constraint model
        model = ConstraintModel()

        # Add all candidates to the model
        all_candidates: list[Candidate] = []
        for _label, candidates in result.candidates.items():
            for candidate in candidates:
                model.add_candidate(candidate)
                all_candidates.append(candidate)

        logger.debug(
            f"  Added {len(all_candidates)} total candidates "
            f"across {len(result.candidates)} labels"
        )

        # Let each classifier declare custom constraints
        for classifier in self.classifiers:
            classifier.declare_constraints(model, result)

        # Auto-generate schema-based constraints
        generator = SchemaConstraintGenerator()
        for classifier in self.classifiers:
            generator.generate_for_classifier(classifier, model, result)

        # Maximize total score (pair each candidate with its score)
        model.maximize([(cand, cand.score) for cand in all_candidates])

        # Solve
        solved, selection = model.solve()

        if not solved:
            logger.warning(
                f"Constraint solver failed for page {result.page_data.page_number}, "
                "falling back to empty selection"
            )
            result.set_solver_selection(frozenset())
            return

        # Mark selected candidates (use frozenset for hashability)
        selected_candidates = frozenset(
            cand for cand in all_candidates if selection.get(id(cand), False)
        )
        logger.debug(
            f"  Solver selected {len(selected_candidates)}/"
            f"{len(all_candidates)} candidates"
        )
        result.set_solver_selection(selected_candidates)

    def _validate_classification_result(self, result: ClassificationResult) -> None:
        """Validate classification invariants and catch programming errors.

        This method runs assertions to verify that the classification process
        produced a consistent and valid result. These checks catch bugs in
        classifier code where elements are incorrectly constructed or tracked.

        Validations performed:
        - All page elements are tracked via candidates (not created directly)
        - All constructed elements appear in the Page hierarchy (no orphans)
        - Element bboxes match the union of source blocks + child elements

        Args:
            result: The classification result to validate

        Raises:
            AssertionError: If any invariant is violated
        """
        # Import here to avoid circular dependency:
        # - classifier.py imports validation.rules
        # - validation.rules imports ClassificationResult from classifier
        # By importing at runtime (inside this method), both modules are fully
        # loaded before the import executes, avoiding the circular import error.
        from build_a_long.pdf_extract.validation.rules import (  # noqa: PLC0415
            assert_constructed_elements_on_page,
            assert_element_bbox_matches_source_and_children,
            assert_no_shared_source_blocks,
            assert_page_elements_tracked,
        )

        assert_page_elements_tracked(result)
        assert_constructed_elements_on_page(result)
        assert_element_bbox_matches_source_and_children(result)
        assert_no_shared_source_blocks(result)
