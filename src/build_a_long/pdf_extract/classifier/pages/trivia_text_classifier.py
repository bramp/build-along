"""
Trivia text classifier.

Purpose
-------
Identify trivia/flavor text blocks on LEGO instruction pages. These are
informational text blocks containing stories, facts, or background
information about the set's theme, not part of the building instructions.

Heuristic
---------
- Look for dense clusters of Text blocks (many blocks close together)
- Text with significant content (many characters)
- Located in a contiguous vertical area
- May have an associated background image

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import BBox, filter_by_max_area
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    TriviaText,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing, Image, Text

log = logging.getLogger(__name__)


class _TriviaTextScore(Score):
    """Internal score representation for trivia text classification."""

    total_characters: int
    """Total number of characters across all text blocks."""

    text_lines: list[str]
    """The text content from all matched blocks."""

    def score(self) -> Weight:
        """Calculate final weighted score from components.

        Score based on character count.
        """
        # Score based on character count (max at 500 chars)
        return min(1.0, self.total_characters / 500.0)


class TriviaTextClassifier(LabelClassifier):
    """Classifier for trivia/flavor text on instruction pages."""

    output = "trivia_text"
    requires = frozenset()  # No dependencies

    @staticmethod
    def _is_trivia_content(text: str) -> bool:
        """Check if text looks like trivia content (has actual words).

        Returns False for:
        - Empty text
        - Short numeric text like "2x", "17", "20"
        - All-digit text like "6234567" (element IDs)
        """
        text = text.strip()
        if not text:
            return False
        # Skip short text that's mostly numeric (labels/counts like "2x", "17")
        if len(text) <= 5 and text.replace("x", "").replace("X", "").isdigit():
            return False
        # Skip text that's all digits (element IDs, part numbers)
        return not text.isdigit()

    def _score(self, result: ClassificationResult) -> None:
        """Find dense clusters of text and create candidates."""
        page_data = result.page_data
        config = self.config.trivia_text

        log.debug(
            "[trivia_text] Checking page %d with %d blocks",
            page_data.page_number,
            len(page_data.blocks),
        )

        # Collect text blocks that look like trivia content (actual words)
        content_blocks: list[Text] = [
            block
            for block in page_data.blocks
            if isinstance(block, Text) and self._is_trivia_content(block.text)
        ]

        if not content_blocks:
            log.debug("[trivia_text] No content text blocks found")
            return

        # Find clusters of spatially close text blocks
        clusters = self._cluster_text_blocks(content_blocks, config.proximity_margin)

        for cluster in clusters:
            # Calculate total characters
            total_chars = sum(len(block.text) for block in cluster)

            # Need minimum characters to be considered trivia
            if total_chars < config.min_character_count:
                log.debug(
                    "[trivia_text] Cluster with %d blocks rejected: "
                    "only %d chars (need %d)",
                    len(cluster),
                    total_chars,
                    config.min_character_count,
                )
                continue

            # Calculate combined bbox from content blocks only
            combined_bbox = BBox.union_all([b.bbox for b in cluster])

            # Find any images/drawings that overlap with the text area
            related_visuals = self._find_related_visuals(
                combined_bbox, page_data.blocks, page_data.bbox
            )

            # Collect text lines
            text_lines = [b.text for b in cluster]

            score_details = _TriviaTextScore(
                total_characters=total_chars,
                text_lines=text_lines,
            )

            combined = score_details.score()

            if combined < config.min_score:
                log.debug(
                    "[trivia_text] Rejected cluster at %s: score=%.2f < min=%.2f",
                    combined_bbox,
                    combined,
                    config.min_score,
                )
                continue

            # Source blocks: only the content text blocks (not images/drawings)
            # This avoids conflicts with other classifiers for nearby elements
            source_blocks: list[Blocks] = list(cluster)

            # Expand bbox to include related visuals for display
            if related_visuals:
                all_bboxes = [combined_bbox] + [v.bbox for v in related_visuals]
                combined_bbox = BBox.union_all(all_bboxes)

            # Clamp bbox to page bounds (some visuals may slightly exceed page)
            combined_bbox = combined_bbox.clip_to(page_data.bbox)

            result.add_candidate(
                Candidate(
                    bbox=combined_bbox,
                    label="trivia_text",
                    score=combined,
                    score_details=score_details,
                    source_blocks=source_blocks,
                ),
            )

            log.debug(
                "[trivia_text] Candidate at %s: %d text blocks, %d chars, "
                "%d visuals, score=%.2f",
                combined_bbox,
                len(cluster),
                total_chars,
                len(related_visuals),
                combined,
            )

    def _cluster_text_blocks(
        self, blocks: list[Text], margin: float
    ) -> list[list[Text]]:
        """Cluster text blocks by spatial proximity.

        Groups blocks whose bounding boxes are within `margin` of each other.
        Uses a simple union-find approach.
        """
        if not blocks:
            return []

        n = len(blocks)
        # Parent array for union-find
        parent = list(range(n))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Check all pairs for proximity
        for i in range(n):
            for j in range(i + 1, n):
                bbox_i = blocks[i].bbox.expand(margin)
                bbox_j = blocks[j].bbox.expand(margin)
                if bbox_i.overlaps(bbox_j):
                    union(i, j)

        # Group by root
        groups: dict[int, list[Text]] = {}
        for i, block in enumerate(blocks):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(block)

        return list(groups.values())

    def _find_related_visuals(
        self, text_bbox: BBox, all_blocks: Sequence[Blocks], page_bbox: BBox
    ) -> list[Image | Drawing]:
        """Find images and drawings that are related to the trivia text area.

        A visual is considered related if it:
        - Significantly overlaps with the text area, or
        - Is contained within an expanded version of the text area
        - Is NOT a large background element covering most of the page
        """
        related: list[Image | Drawing] = []
        expanded_bbox = text_bbox.expand(20.0)  # 20pt margin

        # Calculate page area to identify background elements
        # page_area = page_bbox.area

        candidate_blocks: list[Image | Drawing] = [
            block for block in all_blocks if isinstance(block, Image | Drawing)
        ]

        # Skip large background elements (covering >50% of page)
        filtered_blocks = filter_by_max_area(
            candidate_blocks, max_ratio=0.5, reference_bbox=page_bbox
        )

        for block in filtered_blocks:
            # Check if visual overlaps with or is contained in text area
            if expanded_bbox.contains(block.bbox) or text_bbox.iou(block.bbox) > 0.1:
                related.append(block)

        return related

    def build(self, candidate: Candidate, result: ClassificationResult) -> TriviaText:
        """Construct a TriviaText element from a candidate."""
        detail_score = candidate.score_details
        assert isinstance(detail_score, _TriviaTextScore)

        # Compute bbox as union of source_blocks
        # This ensures the bbox matches source_blocks as required by the assertion
        bbox = BBox.union_all([b.bbox for b in candidate.source_blocks])

        return TriviaText(
            bbox=bbox,
            text_lines=detail_score.text_lines,
        )
