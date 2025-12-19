from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.score import find_best_scoring


@dataclass
class MockScoredItem:
    score: float
    name: str = "item"


def test_find_best_scoring_empty():
    assert find_best_scoring([]) is None


def test_find_best_scoring_single():
    item = MockScoredItem(score=0.5)
    assert find_best_scoring([item]) == item


def test_find_best_scoring_multiple():
    item1 = MockScoredItem(score=0.1, name="low")
    item2 = MockScoredItem(score=0.9, name="high")
    item3 = MockScoredItem(score=0.5, name="mid")

    # Order shouldn't matter
    assert find_best_scoring([item1, item2, item3]) == item2
    assert find_best_scoring([item2, item3, item1]) == item2
    assert find_best_scoring([item3, item1, item2]) == item2


def test_find_best_scoring_tie():
    # Stability of max is not guaranteed by python docs for generic iterables
    # without specific constraints, but for lists it usually returns the first
    # max encountered. However, the function contract just says "the item with
    # the highest score".
    item1 = MockScoredItem(score=0.8, name="first")
    item2 = MockScoredItem(score=0.8, name="second")

    best = find_best_scoring([item1, item2])
    assert best is not None
    assert best.score == 0.8
    assert best in [item1, item2]
