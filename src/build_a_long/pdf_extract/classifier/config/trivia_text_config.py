"""Configuration for trivia text classification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class TriviaTextConfig(BaseModel):
    """Configuration for trivia/flavor text classification.

    Trivia text consists of informational text blocks containing stories,
    facts, or background information about the set's theme. These are not
    part of the building instructions themselves.
    """

    min_score: Weight = Field(
        default=0.5, description="Minimum score threshold for trivia text candidates."
    )

    min_text_block_count: int = Field(
        default=5,
        description=(
            "Minimum number of text blocks to form trivia text. Trivia text typically "
            "spans multiple lines/paragraphs. Single text blocks are unlikely to be trivia."
        ),
    )

    min_character_count: int = Field(
        default=200,
        description=(
            "Minimum total characters across all text blocks. Trivia text has substantial "
            "content. This filters out sparse text areas."
        ),
    )

    proximity_margin: float = Field(
        default=50.0,
        description=(
            "Margin (in points) for clustering text blocks by proximity. Text blocks "
            "within this distance of each other are considered part of the same cluster."
        ),
    )
