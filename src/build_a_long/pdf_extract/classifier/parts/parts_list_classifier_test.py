"""Tests for the parts list classifier."""

from build_a_long.pdf_extract.classifier import ClassificationResult, ClassifierConfig
from build_a_long.pdf_extract.classifier.parts.parts_list_classifier import (
    PartsListClassifier,
)
from build_a_long.pdf_extract.classifier.test_utils import PageBuilder


class TestPartsListClassification:
    """Tests for detecting a parts list drawing."""

    def test_parts_list_with_part_candidates(self, candidate_factory) -> None:
        """Test that a drawing containing parts is classified as a parts list."""
        # 1. Setup Page with PageBuilder
        builder = PageBuilder(width=200, height=300)

        # Drawing that will be the parts list
        builder.add_drawing(x=30, y=100, w=140, h=60, id=10)  # bbox: 30,100,170,160

        # Part contents inside the drawing
        builder.add_text("2x", x=40, y=135, w=15, h=10, id=11)
        builder.add_image(x=40, y=110, w=15, h=15, id=12)

        # Another drawing (noise)
        builder.add_drawing(x=20, y=40, w=160, h=40, id=20)

        page = builder.build()

        # Retrieve blocks for candidate creation
        d1 = page.blocks[0]  # The target parts list drawing
        pc1_text = page.blocks[1]
        pc1_img = page.blocks[2]

        result = ClassificationResult(page_data=page)
        factory = candidate_factory(result)

        # 2. Inject Dependencies (Part Candidates)
        # Create a 'part' candidate inside d1
        # First we need a part_count candidate
        pc_candidate = factory.add_part_count(pc1_text)

        # Then a part candidate linking count and image
        factory.add_part(part_count_candidate=pc_candidate, image_block=pc1_img)

        # 3. Run ONLY the PartsListClassifier
        classifier = PartsListClassifier(config=ClassifierConfig())
        classifier.score(result)

        # 4. Verify
        # Should have found one parts list candidate
        candidates = result.get_candidates("parts_list")
        assert len(candidates) > 0

        # The best candidate should be d1
        # Filter for candidates that use d1 as source
        d1_candidates = [c for c in candidates if d1 in c.source_blocks]
        assert len(d1_candidates) == 1
        parts_list_candidate = d1_candidates[0]

        # Score should be high because it contains parts
        assert parts_list_candidate.score > 0.0

    def test_empty_drawing_not_parts_list(self, candidate_factory) -> None:
        """Test that a drawing with no parts is NOT classified as a parts list."""
        builder = PageBuilder()
        builder.add_drawing(x=10, y=10, w=100, h=100, id=1)
        page = builder.build()

        result = ClassificationResult(page_data=page)
        # No part candidates added

        classifier = PartsListClassifier(config=ClassifierConfig())
        classifier.score(result)

        candidates = result.get_candidates("parts_list")

        # If any candidates are generated, they should have 0 score
        if candidates:
            assert all(c.score == 0.0 for c in candidates)
