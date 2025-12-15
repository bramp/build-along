from typing import Any
from unittest.mock import MagicMock

import pymupdf

from build_a_long.pdf_extract.extractor.pymupdf_types import (
    ImageInfoDict,
    RectLikeTuple,
    TexttraceChar,
    TexttraceSpanDict,
)


class PageBuilder:
    """Builder for creating mock PyMuPDF pages for testing."""

    def __init__(self) -> None:
        self._text_spans: list[dict[str, Any]] = []
        self._images: list[dict[str, Any]] = []
        self._drawings: list[dict[str, Any]] = []
        self._bboxlog: list[tuple[str, tuple[float, float, float, float]]] = []

    def add_text(
        self,
        text: str,
        bbox: RectLikeTuple,
        font: str = "Arial",
        size: float = 12.0,
        seqno: int | None = None,
    ) -> "PageBuilder":
        """Add a text span to the page."""
        self._text_spans.append(
            {
                "text": text,
                "bbox": bbox,
                "font": font,
                "size": size,
                "seqno": seqno if seqno is not None else len(self._text_spans),
            }
        )
        self._bboxlog.append(("fill-text", bbox))
        return self

    def add_image(
        self,
        bbox: RectLikeTuple,
        xref: int,
        number: int,
        width: int = 100,
        height: int = 100,
        image_id: str | None = None,
    ) -> "PageBuilder":
        """Add an image to the page."""
        self._images.append(
            {
                "bbox": bbox,
                "xref": xref,
                "number": number,
                "width": width,
                "height": height,
                "image_id": image_id or f"Im{xref}",
            }
        )
        self._bboxlog.append(("fill-image", bbox))
        return self

    def add_drawing(
        self,
        bbox: RectLikeTuple,
        level: int = 0,
        seqno: int | None = None,
    ) -> "PageBuilder":
        """Add a drawing to the page."""
        self._drawings.append(
            {
                "bbox": bbox,
                "level": level,
                "seqno": seqno,
            }
        )
        # Drawings often don't appear in bboxlog exactly as "fill-path",
        # but for matching logic we might need them. Adding as fill-path for now.
        self._bboxlog.append(("fill-path", bbox))
        return self

    def _make_texttrace(self) -> list[TexttraceSpanDict]:
        """Generate texttrace structure from added text spans."""
        spans: list[TexttraceSpanDict] = []
        for span in self._text_spans:
            bbox = span["bbox"]
            text = span["text"]
            # Generate fake chars
            chars: list[TexttraceChar] = [
                (ord(c), i, (bbox[0], bbox[3]), bbox) for i, c in enumerate(text)
            ]
            spans.append(
                {
                    "bbox": bbox,
                    "font": span["font"],
                    "size": span["size"],
                    "seqno": span["seqno"],
                    "chars": chars,
                    # "text": text, # texttrace doesn't have "text" field directly on span?
                    # Actually, TexttraceSpanDict definition in pymupdf_types.py implies it does NOT have 'text'.
                    # But our make_texttrace_span helper in tests put it there? No, it put 'chars'.
                    # Wait, Extractor._extract_text_blocks_from_texttrace uses span.get("bbox") and Text.from_texttrace_span.
                    # Text.from_texttrace_span likely reconstructs text from chars.
                }
            )
        return spans

    def _make_rawdict(self) -> dict[str, Any]:
        """Generate rawdict structure from added text spans."""
        lines = []
        for span in self._text_spans:
            bbox = span["bbox"]
            text = span["text"]
            chars = [
                {
                    "c": c,
                    "bbox": (
                        bbox[0] + i * 4,
                        bbox[1],
                        bbox[0] + (i + 1) * 4,
                        bbox[3],
                    ),
                }
                for i, c in enumerate(text)
            ]
            lines.append(
                {
                    "spans": [
                        {
                            "bbox": bbox,
                            "font": span["font"],
                            "size": span["size"],
                            "chars": chars,
                            "origin": (bbox[0], bbox[3]),
                        }
                    ],
                    "bbox": bbox,
                }
            )

        return {
            "blocks": [
                {
                    "type": 0,  # text block
                    "lines": lines,
                }
            ]
        }

    def _make_image_info(self) -> list[ImageInfoDict]:
        """Generate get_image_info structure."""
        return [
            {
                "number": img["number"],
                "bbox": img["bbox"],
                "width": img["width"],
                "height": img["height"],
                "colorspace": 3,
                "xres": 96,
                "yres": 96,
                "bpc": 8,
                "size": 1000,
                "transform": (1.0, 0.0, 0.0, 1.0, img["bbox"][0], img["bbox"][1]),
                "xref": img["xref"],
                "digest": b"fake_digest",  # Added because ImageInfoDict might expect it
            }
            for img in self._images
        ]

    def _make_images(self) -> list[tuple]:
        """Generate get_images structure."""
        return [
            (
                img["xref"],
                0,  # smask
                img["width"],
                img["height"],
                8,  # bpc
                3,  # colorspace
                "",
                img["image_id"],
                "DCTDecode",
                0,
            )
            for img in self._images
        ]

    def _make_drawings(self) -> list[dict[str, Any]]:
        """Generate get_drawings structure."""
        return [
            {
                "rect": pymupdf.Rect(d["bbox"]),
                "level": d["level"],
                "seqno": d["seqno"],
                "items": [("re", pymupdf.Rect(d["bbox"]))],  # Dummy item
            }
            for d in self._drawings
        ]

    def build_mock_page(self) -> MagicMock:
        """Build and return a configured MagicMock page."""
        mock_page = MagicMock()
        mock_page.rect = MagicMock(x0=0, y0=0, x1=612, y1=792)
        mock_page.transformation_matrix = pymupdf.Identity

        # Configure get_text
        rawdict = self._make_rawdict()
        texttrace = self._make_texttrace()

        def get_text_side_effect(option="text", flags=None):
            if option == "dict":  # Used by Extractor when use_rawdict=True
                return rawdict
            return ""

        mock_page.get_text.side_effect = get_text_side_effect
        mock_page.get_texttrace.return_value = texttrace

        # Configure image methods
        mock_page.get_bboxlog.return_value = self._bboxlog
        mock_page.get_image_info.return_value = self._make_image_info()
        mock_page.get_images.return_value = self._make_images()

        # Configure drawings
        mock_page.get_drawings.return_value = self._make_drawings()

        return mock_page
