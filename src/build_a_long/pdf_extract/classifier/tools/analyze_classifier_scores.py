#!/usr/bin/env python3
"""Analyze classifier scores for golden fixtures.

This script analyzes the scoring of candidates in the golden fixtures,
showing:
1. Winning candidates and their scores (should be near-perfect)
2. Losing candidates with high scores (might indicate scoring issues)
3. Score distribution statistics per label

This helps identify:
- Classifiers that need score tuning (low-scoring winners)
- Potential false positives (high-scoring losers)
- Scoring patterns across different element types

Usage:
    pants run src/build_a_long/pdf_extract/classifier/tools/analyze_classifier_scores.py
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.classifier.parts.parts_image_classifier import (
    PartImageScore,
)
from build_a_long.pdf_extract.classifier.rule_based_classifier import RuleScore
from build_a_long.pdf_extract.extractor import ExtractionResult
from build_a_long.pdf_extract.fixtures import (
    EXPECTED_FIXTURE_FILES,
    FIXTURES_DIR,
    RAW_FIXTURE_FILES,
    extract_element_id,
    load_classifier_config,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def format_score_details(cand: Candidate) -> str | None:
    """Format score details for display.

    Returns a string description of the score details, or None if no details available.
    """
    details = cand.score_details
    if isinstance(details, PartImageScore):
        return details.summary
    if isinstance(details, RuleScore):
        components = ", ".join(
            f"{name}={value:.3f}" for name, value in details.components.items()
        )
        return f"components: {components}"
    return None


def get_golden_test_fixtures() -> list[str]:
    """Get raw fixture files that have corresponding golden expected files.

    This matches the fixtures used by golden_classifier_test.py.
    """
    # Build set of element IDs that have expected files
    expected_ids = set()
    for expected_file in EXPECTED_FIXTURE_FILES:
        # Expected files are like '6509377_page_013_expected.json'
        # We need to find the matching raw file
        raw_file = expected_file.replace("_expected.json", "_raw.json")
        if raw_file in RAW_FIXTURE_FILES:
            expected_ids.add(raw_file)

    return sorted(expected_ids)


@dataclass
class ScoreAnalysis:
    """Analysis of candidate scores for a label."""

    label: str
    fixture: str
    winning_scores: list[float] = field(default_factory=list)
    losing_scores: list[float] = field(default_factory=list)
    winning_candidates: list[Candidate] = field(default_factory=list)
    # High-scoring losers: tuple of (candidate, failure_reason)
    high_scoring_losers: list[tuple[Candidate, str | None]] = field(
        default_factory=list
    )


def analyze_fixture(fixture_file: str) -> dict[str, ScoreAnalysis]:
    """Analyze a single fixture and return score analysis per label.

    Args:
        fixture_file: Name of the raw fixture file (e.g., '6509377_page_013_raw.json')

    Returns:
        Dictionary mapping label names to their score analysis
    """
    fixture_path = FIXTURES_DIR / fixture_file

    extraction: ExtractionResult = ExtractionResult.model_validate_json(
        fixture_path.read_text()
    )

    if not extraction.pages:
        log.warning(f"No pages in {fixture_file}")
        return {}

    # Load config with hints
    element_id = extract_element_id(fixture_file)
    config = load_classifier_config(element_id)

    page = extraction.pages[0]
    result = classify_elements(page, config)

    analyses: dict[str, ScoreAnalysis] = {}

    # Analyze each label
    for label, candidates in result.get_all_candidates().items():
        analysis = ScoreAnalysis(label=label, fixture=fixture_file)

        for candidate in candidates:
            is_built = result.get_constructed(candidate) is not None
            failure_reason = result.get_failure_reason(candidate)

            if is_built:
                analysis.winning_scores.append(candidate.score)
                analysis.winning_candidates.append(candidate)
            else:
                analysis.losing_scores.append(candidate.score)
                # Track high-scoring losers (score > 0.5)
                if candidate.score > 0.5:
                    analysis.high_scoring_losers.append((candidate, failure_reason))

        if analysis.winning_candidates or analysis.high_scoring_losers:
            analyses[label] = analysis

    return analyses


def main() -> None:
    """Analyze scores across all golden test fixtures."""
    # Get fixtures that have golden expected files (matching golden_classifier_test.py)
    golden_fixtures = get_golden_test_fixtures()
    log.info(f"Found {len(golden_fixtures)} golden test fixtures")

    # Aggregate statistics
    all_analyses: dict[str, list[ScoreAnalysis]] = defaultdict(list)

    for fixture_file in golden_fixtures:
        log.info(f"Analyzing {fixture_file}...")
        analyses = analyze_fixture(fixture_file)

        for label, analysis in analyses.items():
            all_analyses[label].append(analysis)

    print("\n" + "=" * 80)
    print("SCORE ANALYSIS SUMMARY")
    print("=" * 80)

    # Labels with low winning scores (should be improved)
    print("\n" + "-" * 80)
    print("LABELS WITH LOW WINNING SCORES (winners scoring < 0.7)")
    print("-" * 80)

    low_score_labels: dict[str, list[tuple[str, float, Candidate]]] = defaultdict(list)

    for label, analyses in sorted(all_analyses.items()):
        for analysis in analyses:
            for cand in analysis.winning_candidates:
                if cand.score < 0.7:
                    low_score_labels[label].append((analysis.fixture, cand.score, cand))

    if low_score_labels:
        for label, entries in sorted(low_score_labels.items()):
            print(f"\n{label}:")
            for fixture, score, cand in sorted(entries, key=lambda x: x[1]):
                print(f"  {fixture}: score={score:.3f}")
                details = format_score_details(cand)
                if details:
                    print(f"    {details}")
    else:
        print("  None - all winners score >= 0.7")

    # Labels with high-scoring losers (potential false positives)
    print("\n" + "-" * 80)
    print("LABELS WITH HIGH-SCORING LOSERS (losers scoring > 0.5)")
    print("-" * 80)

    high_loser_labels: dict[str, list[tuple[str, float, Candidate, str | None]]] = (
        defaultdict(list)
    )

    for label, analyses in sorted(all_analyses.items()):
        for analysis in analyses:
            for cand, failure_reason in analysis.high_scoring_losers:
                high_loser_labels[label].append(
                    (analysis.fixture, cand.score, cand, failure_reason)
                )

    if high_loser_labels:
        for label, entries in sorted(high_loser_labels.items()):
            print(f"\n{label}:")
            # Show top 10 highest-scoring losers
            for fixture, score, cand, reason in sorted(entries, key=lambda x: -x[1])[
                :10
            ]:
                print(f"  {fixture}: score={score:.3f}")
                if reason:
                    print(f"    failure: {reason[:100]}...")
                details = format_score_details(cand)
                if details:
                    print(f"    {details}")
    else:
        print("  None - no losers score > 0.5")

    # Score distribution per label
    print("\n" + "-" * 80)
    print("SCORE DISTRIBUTION BY LABEL (winners)")
    print("-" * 80)

    for label, analyses in sorted(all_analyses.items()):
        all_winning = []
        for analysis in analyses:
            all_winning.extend(analysis.winning_scores)

        if all_winning:
            min_score = min(all_winning)
            max_score = max(all_winning)
            avg_score = sum(all_winning) / len(all_winning)
            print(
                f"  {label}: n={len(all_winning)}, "
                f"min={min_score:.3f}, avg={avg_score:.3f}, max={max_score:.3f}"
            )


if __name__ == "__main__":
    main()
