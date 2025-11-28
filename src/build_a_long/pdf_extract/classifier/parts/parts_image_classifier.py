"""
Part image classifier.

Purpose
-------
Creates PartImage candidates from Image blocks on the page.
These candidates are then paired with part counts by PartsClassifier.

This classifier simply wraps each Image as a potential part image candidate,
without filtering or dependencies on other classifiers.

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PartImage,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Image,
)

log = logging.getLogger(__name__)


class _PartImageScore(Score):
    """Score details for a part image candidate.

    Attributes:
        image: The source Image block being scored as a potential part image
    """

    image: Image

    def score(self) -> Weight:
        """Return the score value (always 1.0 for part images)."""
        return 1.0


# TODO Should this be called PartImageClassifier instead?
@dataclass(frozen=True)
class PartsImageClassifier(LabelClassifier):
    """Classifier for part images based on size heuristics.

    Filters images to find those that could be part diagrams based on their
    size relative to the page. Typically part images are around 1/10 of the
    page width/height.

    Does NOT pair images with part counts - that's done by PartsClassifier.
    """

    output = "part_image"
    requires = frozenset()  # No dependencies - works on raw Image blocks

    def _score(self, result: ClassificationResult) -> None:
        """Create PartImage candidates from all Image blocks.

        Simply wraps each Image block as a PartImage candidate.
        PartsClassifier will handle pairing with part counts.
        """
        page_data = result.page_data

        # Get all images from the page
        images: list[Image] = [e for e in page_data.blocks if isinstance(e, Image)]

        if not images:
            log.debug(
                "[part_image] No images found on page %s",
                page_data.page_number,
            )
            return

        # Create a PartImage candidate for each Image
        for img in images:
            score_details = _PartImageScore(image=img)

            result.add_candidate(
                Candidate(
                    bbox=img.bbox,
                    label="part_image",
                    score=1.0,
                    score_details=score_details,
                    source_blocks=[img],
                ),
            )

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "[part_image] Created %d part_image candidates on page %s",
                len(images),
                page_data.page_number,
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> PartImage:
        """Construct a PartImage element from a single part_image candidate.

        Args:
            candidate: The part_image candidate to construct
            result: Classification result for context

        Returns:
            PartImage: The constructed part image element
        """
        # Simply create a PartImage with the candidate's bbox
        return PartImage(bbox=candidate.bbox)
