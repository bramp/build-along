"""Configuration for trivia text classification."""

from __future__ import annotations

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.score import Weight


class TriviaTextConfig(BaseModel):
    """Configuration for trivia/flavor text classification.

    Trivia text consists of informational text blocks containing stories,
    facts, or background information about the set's theme. These are not
    part of the building instructions themselves.
    """

    min_score: Weight = 0.5
    """Minimum score threshold for trivia text candidates."""

    min_text_blocks: int = 5
    """Minimum number of text blocks to form trivia text.
    
    Trivia text typically spans multiple lines/paragraphs. Single text
    blocks are unlikely to be trivia.
    """

    min_total_characters: int = 200
    """Minimum total characters across all text blocks.
    
    Trivia text has substantial content. This filters out sparse text areas.
    """

    proximity_margin: float = 50.0
    """Margin (in points) for clustering text blocks by proximity.
    
    Text blocks within this distance of each other are considered part
    of the same cluster.
    """
