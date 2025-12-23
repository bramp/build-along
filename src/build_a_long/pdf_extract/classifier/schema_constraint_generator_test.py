"""Unit tests for SchemaConstraintGenerator.

Tests cover:
- Type introspection (_get_field_element_type, _get_candidate_element_type)
- Type matching (_types_match)
- Field type parsing (_parse_field_type)
- Constraint generation (field constraints, custom constraints)
"""

from collections.abc import Sequence

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.constraint_model import ConstraintModel
from build_a_long.pdf_extract.classifier.schema_constraint_generator import (
    SchemaConstraintGenerator,
)
from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LegoPageElement,
    Part,
    PartCount,
    PartImage,
    PartNumber,
    PartsList,
    Step,
    StepNumber,
)


class MockScore(Score):
    """Simple mock score for testing."""

    def score(self) -> float:
        return 1.0


class TestGetFieldElementType:
    """Tests for _get_field_element_type method.

    This method takes (element_class, field_name) and looks up the field type
    from the Pydantic model.
    """

    def test_required_element_field(self) -> None:
        """Required element field returns the type."""
        generator = SchemaConstraintGenerator()
        # Part.count is PartCount
        result = generator._get_field_element_type(Part, "count")
        assert result is PartCount

    def test_optional_element_field(self) -> None:
        """Optional element field returns the inner type."""
        generator = SchemaConstraintGenerator()
        # Part.number is PartNumber | None
        result = generator._get_field_element_type(Part, "number")
        assert result is PartNumber

    def test_sequence_field(self) -> None:
        """Sequence field returns the inner type."""
        generator = SchemaConstraintGenerator()
        # PartsList.parts is Sequence[Part]
        result = generator._get_field_element_type(PartsList, "parts")
        assert result is Part

    def test_non_element_field(self) -> None:
        """Non-element fields return None."""
        generator = SchemaConstraintGenerator()
        # StepNumber.value is int
        result = generator._get_field_element_type(StepNumber, "value")
        assert result is None

    def test_unknown_field(self) -> None:
        """Unknown field names return None."""
        generator = SchemaConstraintGenerator()
        result = generator._get_field_element_type(Part, "nonexistent")
        assert result is None


class TestGetCandidateElementType:
    """Tests for _get_candidate_element_type method.

    Tests Pydantic generic introspection via __pydantic_generic_metadata__.
    """

    def test_candidate_with_type(self) -> None:
        """Candidate[T] returns T (via Pydantic metadata)."""
        generator = SchemaConstraintGenerator()
        result = generator._get_candidate_element_type(Candidate[Part])
        assert result is Part

    def test_candidate_with_part_count(self) -> None:
        """Candidate[PartCount] returns PartCount."""
        generator = SchemaConstraintGenerator()
        result = generator._get_candidate_element_type(Candidate[PartCount])
        assert result is PartCount

    def test_list_of_candidates(self) -> None:
        """list[Candidate[T]] returns T."""
        generator = SchemaConstraintGenerator()
        result = generator._get_candidate_element_type(list[Candidate[Part]])
        assert result is Part

    def test_optional_candidate(self) -> None:
        """Optional[Candidate[T]] returns T via Union handling."""
        generator = SchemaConstraintGenerator()
        # Use | None syntax which creates types.UnionType
        result = generator._get_candidate_element_type(Candidate[PartCount] | None)
        assert result is PartCount

    def test_plain_candidate_no_type(self) -> None:
        """Candidate without type parameter returns None."""
        generator = SchemaConstraintGenerator()
        # Plain Candidate (no generic) - no pydantic metadata
        result = generator._get_candidate_element_type(Candidate)
        assert result is None

    def test_non_candidate_type(self) -> None:
        """Non-Candidate types return None."""
        generator = SchemaConstraintGenerator()
        assert generator._get_candidate_element_type(Part) is None
        assert generator._get_candidate_element_type(str) is None
        assert generator._get_candidate_element_type(list[str]) is None


class TestTypesMatch:
    """Tests for _types_match method."""

    def test_exact_match(self) -> None:
        """Same types match."""
        generator = SchemaConstraintGenerator()
        assert generator._types_match(Part, Part) is True
        assert generator._types_match(PartCount, PartCount) is True

    def test_subclass_match(self) -> None:
        """Subclass types match superclass."""
        generator = SchemaConstraintGenerator()
        # All concrete elements are subclasses of LegoPageElement
        assert generator._types_match(Part, LegoPageElement) is True
        assert generator._types_match(Step, LegoPageElement) is True

    def test_no_match(self) -> None:
        """Unrelated types don't match."""
        generator = SchemaConstraintGenerator()
        assert generator._types_match(Part, PartCount) is False
        assert generator._types_match(Step, PartsList) is False


class TestParseFieldType:
    """Tests for _parse_field_type method."""

    def test_required_single_element(self) -> None:
        """Required single element field."""
        generator = SchemaConstraintGenerator()
        child_type, cardinality = generator._parse_field_type(StepNumber, True)
        assert child_type is StepNumber
        assert cardinality == "required_one"

    def test_optional_single_element(self) -> None:
        """Optional single element field."""
        generator = SchemaConstraintGenerator()
        child_type, cardinality = generator._parse_field_type(PartNumber | None, False)
        assert child_type is PartNumber
        assert cardinality == "optional_one"

    def test_sequence_field(self) -> None:
        """Sequence field returns 'many' cardinality."""
        generator = SchemaConstraintGenerator()
        child_type, cardinality = generator._parse_field_type(Sequence[Part], True)
        assert child_type is Part
        assert cardinality == "many"

    def test_list_field(self) -> None:
        """List field returns 'many' cardinality."""
        generator = SchemaConstraintGenerator()
        child_type, cardinality = generator._parse_field_type(list[Part], True)
        assert child_type is Part
        assert cardinality == "many"

    def test_non_element_field(self) -> None:
        """Non-element field returns None."""
        generator = SchemaConstraintGenerator()
        child_type, cardinality = generator._parse_field_type(str, True)
        assert child_type is None
        assert cardinality == ""


class TestConstraintModelIntegration:
    """Integration tests with ConstraintModel."""

    def _make_candidate(
        self, label: str, score_val: float = 0.9, score_details: Score | None = None
    ) -> Candidate:
        """Create a test candidate."""
        return Candidate(
            bbox=BBox(0, 0, 10, 10),
            label=label,
            score=score_val,
            score_details=score_details or MockScore(),
        )

    def test_has_candidate_check(self) -> None:
        """Generator should only add constraints for registered candidates."""
        model = ConstraintModel()

        # Create candidates but don't register one
        cand1 = self._make_candidate("part")
        cand2 = self._make_candidate("part")

        model.add_candidate(cand1)
        # cand2 is NOT added

        assert model.has_candidate(cand1) is True
        assert model.has_candidate(cand2) is False

    def test_solver_maximizes_score(self) -> None:
        """Solver should prefer higher-scoring candidates."""
        model = ConstraintModel()

        # Two exclusive candidates with different scores
        cand_high = self._make_candidate("part", score_val=0.9)
        cand_low = self._make_candidate("part", score_val=0.5)

        model.add_candidate(cand_high)
        model.add_candidate(cand_low)

        # Make them mutually exclusive
        model.at_most_one_of([cand_high, cand_low])

        # Maximize by score
        model.maximize(
            [
                (cand_high, int(cand_high.score * 1000)),
                (cand_low, int(cand_low.score * 1000)),
            ]
        )

        success, selection = model.solve()
        assert success is True
        assert selection[cand_high.id] is True
        assert selection[cand_low.id] is False


class TestLabelMapping:
    """Tests for label to element class mapping."""

    def test_get_element_class_known_labels(self) -> None:
        """Known labels map to correct element classes."""
        generator = SchemaConstraintGenerator()
        assert generator._get_element_class("part") is Part
        assert generator._get_element_class("part_count") is PartCount
        assert generator._get_element_class("part_image") is PartImage
        assert generator._get_element_class("step") is Step
        assert generator._get_element_class("step_number") is StepNumber
        assert generator._get_element_class("parts_list") is PartsList

    def test_get_element_class_unknown_label(self) -> None:
        """Unknown labels return None."""
        generator = SchemaConstraintGenerator()
        assert generator._get_element_class("unknown_label") is None
        assert generator._get_element_class("foo") is None
