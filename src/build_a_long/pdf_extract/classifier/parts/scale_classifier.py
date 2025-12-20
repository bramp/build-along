"""
Scale classifier.

Purpose
-------
Identify 1:1 scale indicators that show the actual printed size of a piece.
These typically appear at the bottom of instruction pages, containing a piece
length indicator (number in a circle) and "1:1" text within a bounding box.

Key Characteristics
-------------------
- Contains "1:1" text
- Enclosed in a white rectangular box (Drawing)
- Has a PieceLength indicator inside (number in circle)
- Usually at the bottom of the page

The Scale element helps builders verify piece lengths by measuring directly
against the printed instruction manual.
"""

import logging
from collections.abc import Sequence
from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.extractor.bbox import (
    BBox,
    filter_contained,
    group_by_similar_bbox,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PieceLength,
    Scale,
    ScaleText,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing

log = logging.getLogger(__name__)


class _ScaleScore(Score):
    """Score details for Scale candidates."""

    piece_length_candidate: Candidate | None = None
    """The PieceLength candidate associated with this scale."""

    scale_text_candidate: Candidate
    """The ScaleText candidate associated with this scale."""

    def score(self) -> float:
        """Calculate overall score."""
        # Base score from scale text
        base = self.scale_text_candidate.score

        # Bonus for having piece length inside
        if self.piece_length_candidate:
            base += 0.5

        return min(1.0, base)


class ScaleClassifier(LabelClassifier):
    """Classifier for 1:1 scale indicators."""

    output: ClassVar[str] = "scale"
    requires: ClassVar[frozenset[str]] = frozenset({"piece_length", "scale_text"})

    def _find_containing_box(
        self, inner_bbox: BBox, containers: Sequence[Drawing], margin: float = 10.0
    ) -> Drawing | None:
        """Find the smallest container that contains inner_bbox (with margin)."""
        best_container: Drawing | None = None
        best_area = float("inf")

        for container in containers:
            # Expand container slightly to allow for text sticking out
            expanded = container.bbox.expand(margin)
            if not expanded.contains(inner_bbox):
                continue

            container_area = container.bbox.area
            if container_area < best_area:
                best_area = container_area
                best_container = container

        return best_container

    def _score(self, result: ClassificationResult) -> None:
        """Score potential Scale indicators."""
        page_data = result.page_data

        # Get candidates
        piece_length_candidates = result.get_scored_candidates("piece_length")
        scale_text_candidates = result.get_scored_candidates("scale_text")

        # Get all Drawing blocks for finding containers
        # Filter out large drawings (backgrounds, etc.) that are > 20% of page area
        page_area = page_data.bbox.area
        max_drawing_area = page_area * 0.2
        drawings = [
            b
            for b in page_data.blocks
            if isinstance(b, Drawing) and b.bbox.area <= max_drawing_area
        ]

        # Iterate over 1:1 text candidates
        for scale_text_cand in scale_text_candidates:
            # Find the smallest containing Drawing box
            containing_box = self._find_containing_box(scale_text_cand.bbox, drawings)

            if not containing_box:
                log.debug("[scale] No containing box found, skipping")
                continue

            log.debug(
                "[scale] Found containing box at %s",
                containing_box.bbox,
            )
            search_bbox = containing_box.bbox

            # Look for PieceLength candidates inside the box
            piece_length_candidate = self._find_candidate_in_box(
                search_bbox, piece_length_candidates
            )

            piece_length_block_ids = set()
            if piece_length_candidate:
                log.debug(
                    "[scale] Found piece_length at %s inside box",
                    piece_length_candidate.bbox,
                )
                piece_length_block_ids = {
                    b.id for b in piece_length_candidate.source_blocks
                }

            # Get scale_text block IDs to exclude from scale's source_blocks
            scale_text_block_ids = {b.id for b in scale_text_cand.source_blocks}

            # Blocks owned by children (will be excluded from scale's source_blocks)
            child_block_ids = piece_length_block_ids | scale_text_block_ids

            # Create score
            score = _ScaleScore(
                piece_length_candidate=piece_length_candidate,
                scale_text_candidate=scale_text_cand,
            )

            # Collect source blocks for Scale (container + diagram).
            # ScaleText and PieceLength blocks are owned by their candidates.
            source_blocks: Sequence[Blocks] = []

            # Find similar drawings to the containing box (border/shadow effects)
            similar_groups = group_by_similar_bbox(drawings, tolerance=2.0)
            for group in similar_groups:
                if containing_box in group:
                    for d in group:
                        if d.id not in child_block_ids:
                            source_blocks.append(d)
                    break

            # Capture all drawings inside the box (vector part image, ruler, etc.)
            contained_drawings = filter_contained(drawings, search_bbox)
            for drawing in contained_drawings:
                if drawing not in source_blocks and drawing.id not in child_block_ids:
                    source_blocks.append(drawing)

            # Note: We don't add text blocks here - ScaleText candidate owns them
            # and any shadow effects should ideally be handled there.

            log.debug(
                "[scale] Collected %d source blocks for scale at %s",
                len(source_blocks),
                containing_box.bbox,
            )

            result.add_candidate(
                Candidate(
                    label=self.output,
                    bbox=containing_box.bbox,
                    score=score.score(),
                    score_details=score,
                    source_blocks=source_blocks,
                )
            )

    def _find_candidate_in_box(
        self,
        box_bbox: BBox,
        candidates: Sequence[Candidate],
    ) -> Candidate | None:
        """Find the best candidate inside the given box."""
        # Find candidates contained in the box
        contained = filter_contained(candidates, box_bbox)

        if not contained:
            return None

        # Return the highest-scoring one
        return max(contained, key=lambda c: c.score)

    def build(self, candidate: Candidate, result: ClassificationResult) -> Scale:
        """Construct a Scale element from a candidate."""
        score = candidate.score_details
        assert isinstance(score, _ScaleScore)

        # Build the PieceLength if we have one
        piece_length: PieceLength | None = None
        if score.piece_length_candidate:
            pl_elem = result.build(score.piece_length_candidate)
            assert isinstance(pl_elem, PieceLength)
            piece_length = pl_elem

        # If no piece length found, we can't build a valid Scale
        if piece_length is None:
            raise ValueError("Scale requires a PieceLength element")

        # Build ScaleText
        scale_text = result.build(score.scale_text_candidate)
        assert isinstance(scale_text, ScaleText)

        # Note: Scale diagrams are typically vector drawings, not Images.
        # The DiagramClassifier works with Images, so we don't expect to find
        # a Diagram candidate here. The drawing blocks are owned by the Scale
        # via source_blocks but we don't create a separate child element for them.
        # In the future, we could add a VectorDiagram element if needed.

        # The final bbox should be the union of all components
        bboxes = [b.bbox for b in candidate.source_blocks]
        bboxes.append(piece_length.bbox)
        bboxes.append(scale_text.bbox)
        final_bbox = BBox.union_all(bboxes)

        return Scale(
            bbox=final_bbox,
            length=piece_length,
            text=scale_text,
            diagram=None,  # No Image-based diagram in Scales (they use Drawings)
        )
