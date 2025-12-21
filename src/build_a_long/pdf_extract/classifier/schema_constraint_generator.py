"""Schema-based constraint generation for classification.

This module analyzes Pydantic element schemas and score objects to automatically
generate constraints for the CP-SAT solver.

Design Decisions:
-----------------
1. Use EXPLICIT field naming convention to identify dependencies:
   - Fields ending in '_candidate' are single dependencies
   - Fields ending in '_candidates' are list dependencies
   This makes dependencies obvious in code (no hidden magic).

2. Use Pydantic's model_fields to introspect schema structure.

3. Support __constraint_rules__ class variable for custom constraints beyond
   what can be inferred from field types.

Example:
    class Step(LegoPageElement):
        # Explicit naming makes dependencies clear
        step_number: StepNumber  # Required field
        diagram: Diagram | None  # Optional field

        # Score references candidates explicitly
        class _StepScore(Score):
            step_number_candidate: Candidate  # "_candidate" suffix = dependency
            parts_list_candidate: Candidate | None
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, get_args, get_origin

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Arrow,
    Background,
    BagNumber,
    Decoration,
    Diagram,
    Divider,
    LegoPageElement,
    LoosePartSymbol,
    OpenBag,
    PageNumber,
    Part,
    PartCount,
    PartImage,
    PartNumber,
    PartsList,
    PieceLength,
    Preview,
    ProgressBar,
    ProgressBarBar,
    ProgressBarIndicator,
    RotationSymbol,
    Scale,
    ScaleText,
    Shine,
    Step,
    StepCount,
    StepNumber,
    SubAssembly,
    SubStep,
    TriviaText,
)

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classification_result import (
        ClassificationResult,
    )
    from build_a_long.pdf_extract.classifier.constraint_model import ConstraintModel
    from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier

log = logging.getLogger(__name__)

# Map label strings to element classes
LABEL_TO_ELEMENT_CLASS: dict[str, type[LegoPageElement]] = {
    "page_number": PageNumber,
    "step_number": StepNumber,
    "step_count": StepCount,
    "part_count": PartCount,
    "part_number": PartNumber,
    "piece_length": PieceLength,
    "part_image": PartImage,
    "shine": Shine,
    "scale": Scale,
    "scale_text": ScaleText,
    "progress_bar": ProgressBar,
    "progress_bar_bar": ProgressBarBar,
    "progress_bar_indicator": ProgressBarIndicator,
    "background": Background,
    "trivia_text": TriviaText,
    "decoration": Decoration,
    "divider": Divider,
    "rotation_symbol": RotationSymbol,
    "arrow": Arrow,
    "part": Part,
    "parts_list": PartsList,
    "bag_number": BagNumber,
    "open_bag": OpenBag,
    "loose_part_symbol": LoosePartSymbol,
    "diagram": Diagram,
    "preview": Preview,
    "substep": SubStep,
    "subassembly": SubAssembly,
    "step": Step,
}


class SchemaConstraintGenerator:
    """Generates CP-SAT constraints from Pydantic schemas and score objects.

    This class analyzes element schemas to automatically generate constraints
    like "if parent selected, child must be selected" based on field types.

    Convention: Dependencies are identified by field naming:
    - score_details.some_field_candidate: Candidate → single dependency
    - score_details.some_field_candidates: list[Candidate] → multiple dependencies
    """

    def generate_for_classifier(
        self,
        classifier: LabelClassifier,
        model: ConstraintModel,
        result: ClassificationResult,
    ) -> None:
        """Auto-generate constraints for a classifier's output type.

        Args:
            classifier: The classifier to generate constraints for
            model: The constraint model to add constraints to
            result: The classification result containing candidates
        """
        element_class = self._get_element_class(classifier.output)
        if not element_class:
            log.debug(
                "No element class found for label '%s', skipping auto-generation",
                classifier.output,
            )
            return

        candidates_seq = result.get_scored_candidates(classifier.output)
        if not candidates_seq:
            log.debug(
                "No candidates for label '%s', skipping constraint generation",
                classifier.output,
            )
            return

        # Convert Sequence to list for internal use
        candidates = list(candidates_seq)

        log.debug(
            "Auto-generating constraints for %s (%d candidates)",
            element_class.__name__,
            len(candidates),
        )

        # Generate structural constraints from Pydantic fields
        self._generate_field_constraints(element_class, candidates, model, result)

        # Generate custom constraints from __constraint_rules__
        self._generate_custom_constraints(element_class, candidates, model, result)

    def _get_element_class(self, label: str) -> type[LegoPageElement] | None:
        """Map label to element class.

        Args:
            label: The label string (e.g., 'step', 'parts_list')

        Returns:
            The corresponding element class, or None if not found
        """
        return LABEL_TO_ELEMENT_CLASS.get(label)

    def _generate_field_constraints(
        self,
        element_class: type[LegoPageElement],
        candidates: list[Candidate],
        model: ConstraintModel,
        result: ClassificationResult,
    ) -> None:
        """Generate constraints from Pydantic field types.

        For each field that references another element:
        - Required fields → parent selected ⇒ exactly one child selected
        - Optional fields → parent selected ⇒ at most one child selected
        - Sequence fields → zero or more allowed (no constraint)

        Args:
            element_class: The Pydantic model class to analyze
            candidates: Candidates for this element type
            model: The constraint model
            result: The classification result
        """
        for field_name, field_info in element_class.model_fields.items():
            # Skip special fields
            if field_name in ("tag", "bbox"):
                continue

            field_type = field_info.annotation
            is_required = field_info.is_required()

            # Analyze field type
            child_type, cardinality = self._parse_field_type(field_type, is_required)

            if not child_type:
                continue  # Not a LegoPageElement field

            # Generate constraints for each candidate
            for parent_cand in candidates:
                self._add_field_constraint(
                    parent_cand, field_name, child_type, cardinality, model, result
                )

    def _parse_field_type(
        self, field_type: Any, is_required: bool
    ) -> tuple[type[LegoPageElement] | None, str]:
        """Parse a field type to determine child type and cardinality.

        Args:
            field_type: The field's annotation
            is_required: Whether the field is required (no default)

        Returns:
            Tuple of (child_type, cardinality) where cardinality is:
            - 'required_one': Must have exactly one child
            - 'optional_one': May have at most one child
            - 'many': May have zero or more children
            - None if not a LegoPageElement field
        """
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Optional[X] / X | None
        if origin is type(None) or (args and type(None) in args):
            # Union with None = optional
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                return self._parse_field_type(non_none_types[0], False)

        # Handle Sequence[X]
        if origin in (list, tuple, type(list), type(tuple)) and args and len(args) > 0:
            child_type = args[0]
            if isinstance(child_type, type) and issubclass(child_type, LegoPageElement):
                return child_type, "many"

        # Handle single element
        if isinstance(field_type, type) and issubclass(field_type, LegoPageElement):
            cardinality = "required_one" if is_required else "optional_one"
            return field_type, cardinality

        return None, ""

    def _add_field_constraint(
        self,
        parent_cand: Candidate,
        field_name: str,
        child_type: type[LegoPageElement],
        cardinality: str,
        model: ConstraintModel,
        result: ClassificationResult,
    ) -> None:
        """Add constraint for a specific field relationship.

        Extracts child candidates from parent's score_details using naming convention:
        - field_name + '_candidate' for single child
        - field_name + '_candidates' for multiple children

        Args:
            parent_cand: The parent candidate
            field_name: Name of the field in the element class
            child_type: Type of child element
            cardinality: Constraint cardinality
            model: The constraint model
            result: The classification result
        """
        # Extract child candidates from score_details using naming convention
        child_candidates = self._extract_child_candidates(parent_cand, field_name)

        if not child_candidates:
            return  # No children to constrain

        parent_var = model.get_var(parent_cand)
        child_vars = [model.get_var(c) for c in child_candidates]

        if cardinality == "required_one":
            # If parent selected, exactly one child must be selected
            model.model.Add(sum(child_vars) == 1).OnlyEnforceIf(parent_var)
            log.debug(
                "  Field '%s': required_one (%d child candidates)",
                field_name,
                len(child_candidates),
            )

        elif cardinality == "optional_one":
            # If parent selected, at most one child
            model.model.Add(sum(child_vars) <= 1).OnlyEnforceIf(parent_var)
            log.debug(
                "  Field '%s': optional_one (%d child candidates)",
                field_name,
                len(child_candidates),
            )

        elif cardinality == "many":
            # Sequence: zero or more allowed (no constraint needed)
            # But we can add: if parent selected, at least one child often makes sense
            # Skip for now - let custom rules handle this
            pass

    def _extract_child_candidates(
        self, parent_cand: Candidate, field_name: str
    ) -> list[Candidate]:
        """Extract child candidates from parent's score_details.

        Uses EXPLICIT naming convention:
        - Looks for score_details.{field_name}_candidate (single)
        - Looks for score_details.{field_name}_candidates (list)

        This makes dependencies obvious in code.

        Args:
            parent_cand: The parent candidate
            field_name: The field name (e.g., 'step_number', 'parts_list')

        Returns:
            List of child candidates (empty if none found)
        """
        score_details = parent_cand.score_details
        if not score_details:
            return []

        children = []

        # Try single: {field_name}_candidate
        single_attr = f"{field_name}_candidate"
        if hasattr(score_details, single_attr):
            value = getattr(score_details, single_attr)
            if isinstance(value, Candidate):
                children.append(value)

        # Try multiple: {field_name}_candidates
        multi_attr = f"{field_name}_candidates"
        if hasattr(score_details, multi_attr):
            value = getattr(score_details, multi_attr)
            if isinstance(value, list):
                children.extend([c for c in value if isinstance(c, Candidate)])

        return children

    def _generate_custom_constraints(
        self,
        element_class: type[LegoPageElement],
        candidates: list[Candidate],
        model: ConstraintModel,
        result: ClassificationResult,
    ) -> None:
        """Generate custom constraints from __constraint_rules__.

        This handles constraints that can't be inferred from field types,
        such as uniqueness, no-orphans, spatial relationships, etc.

        Args:
            element_class: The element class to check for rules
            candidates: Candidates for this element type
            model: The constraint model
            result: The classification result
        """
        if not hasattr(element_class, "__constraint_rules__"):
            return

        rules = element_class.__constraint_rules__  # type: ignore[attr-defined]
        log.debug(
            "Processing %d custom constraint rules for %s",
            len(rules),
            element_class.__name__,
        )

        for field_name, rule_config in rules.items():
            self._apply_custom_rule(
                element_class, field_name, rule_config, candidates, model, result
            )

    def _apply_custom_rule(
        self,
        element_class: type[LegoPageElement],
        field_name: str,
        rule_config: dict[str, Any],
        candidates: list[Candidate],
        model: ConstraintModel,
        result: ClassificationResult,
    ) -> None:
        """Apply a single custom constraint rule.

        Supported rules:
        - 'unique_by': Ensure field.attr is unique across all candidates
        - 'min_count': Minimum number of children required
        - 'no_orphans': This element type needs a parent
        - 'assignment': Defer to post-solve spatial assignment

        Args:
            element_class: The element class
            field_name: Field this rule applies to
            rule_config: Rule configuration dict
            candidates: Candidates to apply rule to
            model: The constraint model
            result: The classification result
        """
        # Uniqueness constraint
        if "unique_by" in rule_config:
            self._add_uniqueness_constraint(
                field_name, rule_config["unique_by"], candidates, model
            )

        # Minimum count constraint
        if "min_count" in rule_config:
            self._add_min_count_constraint(
                field_name, rule_config["min_count"], candidates, model
            )

        # No orphans constraint
        if rule_config.get("no_orphans"):
            self._add_no_orphans_constraint(
                element_class.__name__, candidates, model, result
            )

        # Spatial assignment marker (handled post-solve)
        if rule_config.get("assignment") == "spatial":
            log.debug(
                "  Field '%s': spatial assignment (deferred to post-solve)", field_name
            )

    def _add_uniqueness_constraint(
        self,
        field_name: str,
        value_attr: str,
        candidates: list[Candidate],
        model: ConstraintModel,
    ) -> None:
        """Add constraint: field.value_attr must be unique.

        Example: For Step, step_number.value must be unique

        Args:
            field_name: Field name (e.g., 'step_number')
            value_attr: Attribute to check uniqueness on (e.g., 'value')
            candidates: Candidates to constrain
            model: The constraint model
        """
        # Group candidates by value
        by_value: dict[Any, list[Candidate]] = {}

        for cand in candidates:
            # Extract child candidate using naming convention
            children = self._extract_child_candidates(cand, field_name)
            if not children:
                continue

            # Get the value from the child's score or constructed element
            # For now, just use score_details
            child = children[0]
            if hasattr(child.score_details, value_attr):
                value = getattr(child.score_details, value_attr)
                if value not in by_value:
                    by_value[value] = []
                by_value[value].append(cand)

        # Add at_most_one constraint for each value
        for value, cands in by_value.items():
            if len(cands) > 1:
                model.at_most_one_of(cands)
                log.debug(
                    "  Uniqueness: at most one %s with %s=%s (%d candidates)",
                    field_name,
                    value_attr,
                    value,
                    len(cands),
                )

    def _add_min_count_constraint(
        self,
        field_name: str,
        min_count: int,
        candidates: list[Candidate],
        model: ConstraintModel,
    ) -> None:
        """Add constraint: if parent selected, at least min_count children.

        Args:
            field_name: Field name
            min_count: Minimum number of children
            candidates: Parent candidates
            model: The constraint model
        """
        for parent_cand in candidates:
            children = self._extract_child_candidates(parent_cand, field_name)
            if not children:
                continue

            parent_var = model.get_var(parent_cand)
            child_vars = [model.get_var(c) for c in children]

            # If parent selected, sum(children) >= min_count
            model.model.Add(sum(child_vars) >= min_count).OnlyEnforceIf(parent_var)

            log.debug(
                "  Min count: if parent selected, >= %d %s required",
                min_count,
                field_name,
            )

    def _add_no_orphans_constraint(
        self,
        element_name: str,
        candidates: list[Candidate],
        model: ConstraintModel,
        result: ClassificationResult,
    ) -> None:
        """Add constraint: these elements need a parent.

        If any of these candidates selected, at least one parent type must exist.
        For example: arrows need steps, rotation symbols need steps.

        Args:
            element_name: Name of element type needing parent
            candidates: Candidates that need parent
            model: The constraint model
            result: The classification result
        """
        # This is tricky - we need to know what the parent type is
        # For now, log and let classifiers handle this in declare_constraints()
        log.debug(
            "  No orphans: %s elements need parent (handled in classifier)",
            element_name,
        )
