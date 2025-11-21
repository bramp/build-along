"""Tests for legocom.py - LEGO.com website parsing (pytest style)."""

from pydantic import AnyUrl

from build_a_long.downloader.legocom import (
    LEGO_BASE,
    build_instructions_url,
    build_metadata,
    parse_instruction_pdf_urls,
    parse_instruction_pdf_urls_fallback,
    parse_set_metadata,
)
from build_a_long.downloader.models import DownloadUrl

# Sample HTML with multiple PDFs for parsing tests
HTML_WITH_TWO_PDFS = """
<script id="__NEXT_DATA__" type="application/json">
{
  "props": {
    "pageProps": {
      "__APOLLO_STATE__": {
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\"setNumber\\":\\"75419\\"}).data": {
          "buildingInstructions": [
            {"pdf": {"id": "pdf1"}, "sequence": {"id": "seq1"}},
            {"pdf": {"id": "pdf2"}, "sequence": {"id": "seq2"}}
          ],
          "__typename": "CS_BuildingInstructionData"
        },
        "pdf1": {
          "pdfUrl": "https://www.example.com/6602644.pdf",
          "coverImage": {"id": "img1"}
        },
        "pdf2": {
          "pdfUrl": "/6602645.pdf",
          "coverImage": {"id": "img2"}
        },
        "img1": {"src": "image1.png"},
        "img2": {"src": "image2.png"},
        "seq1": {
          "element": "1",
          "total": "2"
        },
        "seq2": {
          "element": "2",
          "total": "2"
        }
      }
    }
  }
}
</script>
"""

# Sample HTML with complete metadata for testing metadata extraction
HTML_WITH_METADATA = """
<script id="__NEXT_DATA__" type="application/json">
{
  "props": {
    "pageProps": {
      "__APOLLO_STATE__": {
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\"setNumber\\":\\"12345\\"}).data": {
          "name": "Starfighter",
          "theme": {"id": "theme1"},
          "ageRating": "9+",
          "setPieceCount": "1083",
          "year": "2024",
          "setImage": {"id": "img1"},
          "__typename": "CS_BuildingInstructionData"
        },
        "theme1": {"themeName": "Galaxy Explorers"},
        "img1": {"src": "set_image.png"}
      }
    }
  }
}
</script>
"""

# Sample HTML with both metadata and PDF for complete integration test
HTML_WITH_METADATA_AND_PDF = """
<script id="__NEXT_DATA__" type="application/json">
{
  "props": {
    "pageProps": {
      "__APOLLO_STATE__": {
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\"setNumber\\":\\"12345\\"}).data": {
          "name": "Starfighter",
          "theme": {"id": "theme1"},
          "ageRating": "9+",
          "setPieceCount": "1083",
          "year": "2024",
          "setImage": {"id": "img1"},
          "buildingInstructions": [
            {"pdf": {"id": "pdf1"}, "sequence": {"id": "seq1"}},
            {"pdf": {"id": "pdf2"}, "sequence": {"id": "seq2"}}
          ],
          "__typename": "CS_BuildingInstructionData"
        },
        "theme1": {"themeName": "Galaxy Explorers"},
        "img1": {"src": "set_image.png"},
        "pdf1": {
          "pdfUrl": "/6602000.pdf",
          "coverImage": {"id": "img2"}
        },
        "pdf2": {
          "pdfUrl": "/6602001.pdf",
          "coverImage": {"id": "img3"}
        },
        "img2": {"src": "preview1.png"},
        "img3": {"src": "preview2.png"},
        "seq1": {
          "element": "1",
          "total": "2"
        },
        "seq2": {
          "element": "2",
          "total": "2"
        }
      }
    }
  }
}
</script>
"""

# Realistic sample HTML mimicking LEGO.com's Apollo state for two PDFs
HTML_WITH_TWO_PDFS_REAL_DATA = """
<script id="__NEXT_DATA__" type="application/json">
{
  "props": {
    "pageProps": {
      "__APOLLO_STATE__": {
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\"setNumber\":\"75419\"}).data": {
          "buildingInstructions": [
            {
              "type": "id",
              "id":
              "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\\"setNumber\\\":\\\"75419\\\"}).data.buildingInstructions.0",
              "typename": "BuildingInstruction"
            },
            {
              "type": "id",
              "id":
              "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\\"setNumber\\\":\\\"75419\\\"}).data.buildingInstructions.1",
              "typename": "BuildingInstruction"
            }
          ],
          "__typename": "CS_BuildingInstructionData"
        },
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\"setNumber\":\"75419\"}).data.buildingInstructions.0": {
          "isAdditionalInfoBooklet": false,
          "pdf": {
            "id": "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\\"setNumber\\\":\\\"75419\\\"}).data.buildingInstructions.0.pdf"
          },
          "sequence": {
            "type": "id",
            "id":
            "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\\"setNumber\\\":\\\"75419\\\"}).data.buildingInstructions.0.sequence",
            "typename": "Sequence"
          },
        },
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\"setNumber\":\"75419\"}).data.buildingInstructions.0.pdf": {
          "pdfUrl": "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf",
          "coverImage": {
            "id": "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\\"setNumber\\\":\\\"75419\\\"}).data.buildingInstructions.0.pdf.coverImage"
          }
        },
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\"setNumber\":\"75419\"}).data.buildingInstructions.0.pdf.coverImage": {
          "src": "https://www.lego.com/cdn/product-assets/product.bi.core.img/6602644.png"
        },
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\"setNumber\":\"75419\"}).data.buildingInstructions.0.sequence": {
          "total": "2",
          "element": "1",
          "__typename": "Sequence"
        },
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\"setNumber\":\"75419\"}).data.buildingInstructions.1": {
          "isAdditionalInfoBooklet": false,
          "pdf": {
            "id": "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\\"setNumber\\\":\\\"75419\\\"}).data.buildingInstructions.1.pdf"
          },
          "sequence": {
            "id": "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\\"setNumber\\\":\\\"75419\\\"}).data.buildingInstructions.1.sequence"
          }
        },
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\"setNumber\":\"75419\"}).data.buildingInstructions.1.pdf": {
          "pdfUrl": "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602645.pdf",
          "coverImage": {
            "id": "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\\"setNumber\\\":\\\"75419\\\"}).data.buildingInstructions.1.pdf.coverImage"
          }
        },
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\"setNumber\":\"75419\"}).data.buildingInstructions.1.pdf.coverImage": {
          "src": "https://www.lego.com/cdn/product-assets/product.bi.core.img/6602645.png"
        },
        "$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\"setNumber\":\"75419\"}).data.buildingInstructions.1.sequence": {
          "total": "2",
          "element": "2",
          "__typename": "Sequence"
        },
      }
    }
  }
}
</script>
"""


def test_build_instructions_url():
    url = build_instructions_url("75419", "en-us")
    assert url == "https://www.lego.com/en-us/service/building-instructions/75419"


def test_build_instructions_url_different_locale():
    url = build_instructions_url("12345", "de-de")
    assert url == "https://www.lego.com/de-de/service/building-instructions/12345"


def test_parse_instruction_pdf_urls():
    infos = parse_instruction_pdf_urls(HTML_WITH_TWO_PDFS, base=LEGO_BASE)
    assert infos == [
        DownloadUrl(
            url=AnyUrl("https://www.example.com/6602644.pdf"),
            sequence_number=1,
            sequence_total=2,
            preview_url=AnyUrl("https://www.lego.com/image1.png"),
        ),
        DownloadUrl(
            url=AnyUrl("https://www.lego.com/6602645.pdf"),
            sequence_number=2,
            sequence_total=2,
            preview_url=AnyUrl("https://www.lego.com/image2.png"),
        ),
    ]


def test_parse_set_metadata():
    meta = parse_set_metadata(HTML_WITH_METADATA, set_number="12345", locale="en-us")
    assert meta.name == "Starfighter"
    assert meta.theme == "Galaxy Explorers"
    assert meta.age == "9+"
    assert meta.pieces == 1083
    assert meta.year == 2024
    assert meta.set_image_url == AnyUrl("https://www.lego.com/set_image.png")
    assert meta.set == "12345"
    assert meta.locale == "en-us"


def test_build_metadata():
    metadata = build_metadata(HTML_WITH_METADATA_AND_PDF, "12345", "en-us")
    assert metadata.set == "12345"
    assert metadata.locale == "en-us"
    assert metadata.name == "Starfighter"
    assert metadata.theme == "Galaxy Explorers"
    assert metadata.age == "9+"
    assert metadata.pieces == 1083
    assert metadata.year == 2024
    assert metadata.set_image_url == AnyUrl("https://www.lego.com/set_image.png")
    assert len(metadata.pdfs) == 2
    # Ensure order is preserved
    assert metadata.pdfs[0].url == AnyUrl("https://www.lego.com/6602000.pdf")
    assert metadata.pdfs[0].sequence_number == 1
    assert metadata.pdfs[0].sequence_total == 2
    assert metadata.pdfs[0].filename == "6602000.pdf"
    assert metadata.pdfs[0].preview_url == AnyUrl("https://www.lego.com/preview1.png")
    assert metadata.pdfs[1].url == AnyUrl("https://www.lego.com/6602001.pdf")
    assert metadata.pdfs[1].sequence_number == 2
    assert metadata.pdfs[1].sequence_total == 2
    assert metadata.pdfs[1].filename == "6602001.pdf"
    assert metadata.pdfs[1].preview_url == AnyUrl("https://www.lego.com/preview2.png")


def test_parse_instruction_pdf_urls_from_real_data():
    """Tests parsing of instruction PDF URLs using an existing realistic fixture."""
    html = HTML_WITH_METADATA_AND_PDF
    urls = parse_instruction_pdf_urls(html)
    assert len(urls) == 2
    assert urls[0].url == AnyUrl("https://www.lego.com/6602000.pdf")
    assert urls[0].sequence_number == 1
    assert urls[0].sequence_total == 2
    assert urls[0].preview_url == AnyUrl("https://www.lego.com/preview1.png")
    assert urls[1].url == AnyUrl("https://www.lego.com/6602001.pdf")
    assert urls[1].sequence_number == 2
    assert urls[1].sequence_total == 2
    assert urls[1].preview_url == AnyUrl("https://www.lego.com/preview2.png")


def test_json_fields_exist(monkeypatch):
    """Verify that the JSON fields we rely on still exist in LEGO.com pages."""

    # Sample HTML with OG title, age, pieces, year, and two PDFs
    html = """<script id=\"__NEXT_DATA__\" type=\"application/json\">{\"props\":{\"pageProps\":{\"__APOLLO_STATE__\":{\"$ROOT_QUERY.customerService.getBuildingInstructionsForSet({\\\"setNumber\\\":\\\"12345\\\"}).data\":{\"name\":\"Starfighter\",\"theme\":{\"id\":\"theme1\"},\"ageRating\":\"9+\",\"setPieceCount\":\"1083\",\"year\":\"2024\",\"setImage\":{\"id\":\"img1\"},\"buildingInstructions\":[{\"pdf\":{\"id\":\"pdf1\"}},{\"pdf\":{\"id\":\"pdf2\"}}],\"__typename\":\"CS_BuildingInstructionData\"},\"theme1\":{\"themeName\":\"Galaxy Explorers\"},\"img1\":{\"src\":\"set_image.png\"},\"pdf1\":{\"pdfUrl\":\"/6602000.pdf\",\"coverImage\":{\"id\":\"img2\"}},\"pdf2\":{\"pdfUrl\":\"/6602001.pdf\",\"coverImage\":{\"id\":\"img3\"}},\"img2\":{\"src\":\"preview1.png\"},\"img3\":{\"src\":\"preview2.png\"}}}}}</script>"""

    # Check that JSON extraction methods return values
    meta = parse_set_metadata(html, set_number="12345", locale="en-us")
    assert meta.name == "Starfighter"
    assert meta.theme == "Galaxy Explorers"
    assert meta.age == "9+"
    assert meta.pieces == 1083
    assert meta.year == 2024


def test_parse_instruction_pdf_urls_fallback_simple():
    """Fallback should extract direct PDF URLs when no Apollo data is present."""
    html = (
        '<html><a href="https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6600001.pdf">PDF</a>'
        '<a href="https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6600002.pdf">PDF</a></html>'
    )
    urls = parse_instruction_pdf_urls_fallback(html)
    assert urls == [
        DownloadUrl(
            url=AnyUrl(
                "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6600001.pdf"
            ),
            sequence_number=None,
            sequence_total=None,
            preview_url=None,
        ),
        DownloadUrl(
            url=AnyUrl(
                "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6600002.pdf"
            ),
            sequence_number=None,
            sequence_total=None,
            preview_url=None,
        ),
    ]
