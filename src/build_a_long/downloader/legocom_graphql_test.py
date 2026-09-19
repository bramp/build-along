"""Tests for GraphQL parsing and fetching."""

from pydantic import AnyUrl

from build_a_long.downloader.legocom_graphql import parse_metadata_from_graphql

SAMPLE_GRAPHQL_DATA = {
    "customerService": {
        "getBuildingInstructionsForSet": {
            "status": "ok",
            "data": {
                "setNumber": "75419",
                "name": "Death Star™",
                "year": "2025",
                "ageRating": "18+",
                "setPieceCount": "9023",
                "theme": {
                    "themeName": "LEGO® Star Wars™",
                },
                "setImage": {
                    "src": "https://www.lego.com/cdn/product-assets/product.img.pri/75419_Prod.png",
                    "alt": "Death Star™",
                },
                "buildingInstructions": [
                    {
                        "isAdditionalInfoBooklet": False,
                        "sequence": {"element": "1", "total": "2"},
                        "pdf": {
                            "pdfUrl": "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf",
                            "fileSize": "92919030",
                            "coverImage": {
                                "src": "https://www.lego.com/cdn/product-assets/product.bi.core.img/6602644.png",
                                "alt": "Booklet 1",
                            },
                        },
                    },
                    {
                        "isAdditionalInfoBooklet": False,
                        "sequence": {"element": "2", "total": "2"},
                        "pdf": {
                            "pdfUrl": "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602645.pdf",
                            "fileSize": "64575790",
                            "coverImage": {
                                "src": "https://www.lego.com/cdn/product-assets/product.bi.core.img/6602645.png",
                                "alt": "Booklet 2",
                            },
                        },
                    },
                ],
            },
        }
    },
    "product": {
        "id": "prod-123",
        "productCode": "75419",
        "name": "Death Star™",
        "slug": "death-star-75419",
        "metaTitle": "Death Star™ 75419",
        "metaDescription": "The ultimate Death Star buildable model",
        "description": "<p>Build the ultimate Death Star!</p>",
        "featuresText": "<ul><li>38 characters</li></ul>",
        "primaryImage": "https://www.lego.com/cdn/cs/set/assets/blt/75419_Prod.png",
        "hires": "https://www.lego.com/cdn/cs/set/assets/blt/75419_Prod.png?width=1500",
        "thumbnail": "https://www.lego.com/cdn/cs/set/assets/blt/75419_Prod.png?width=320",
        "productCategories": [
            {"id": "cat-1", "name": "Star Wars™", "slug": "star-wars"},
            {"id": "cat-2", "name": "Adults Welcome", "slug": "adults-welcome"},
        ],
        "brandCategory": {
            "id": "brand-1",
            "name": "Star Wars™",
            "slug": None,
        },
        "programCategory": None,
        "productMediaAssets": [
            {
                "__typename": "ProductAssetImage",
                "id": "asset-1",
                "url": "https://www.lego.com/asset1.png",
                "altText": "Front View",
            },
            {
                "__typename": "ProductAssetVideo",
                "id": "asset-2",
                "video": {
                    "title": "Designer Video",
                    "description": "Meet the designers",
                    "videoFormats": [
                        {"url": "https://www.lego.com/video.mp4", "quality": "Highest"}
                    ],
                },
            },
        ],
    },
}


def test_parse_metadata_from_graphql():
    meta = parse_metadata_from_graphql(
        SAMPLE_GRAPHQL_DATA,
        set_number="75419",
        locale="en-us",
    )
    assert meta is not None
    assert meta.set == "75419"
    assert meta.name == "Death Star™"
    assert meta.theme == "LEGO® Star Wars™"
    assert meta.pieces == 9023
    assert meta.year == 2025
    assert meta.age == "18+"
    assert meta.description == "<p>Build the ultimate Death Star!</p>"
    assert meta.features_text == "<ul><li>38 characters</li></ul>"
    assert meta.meta_description == "The ultimate Death Star buildable model"
    assert meta.meta_title == "Death Star™ 75419"
    assert meta.slug == "death-star-75419"
    assert meta.brand == "Star Wars™"
    assert meta.categories == ["Star Wars™", "Adults Welcome"]
    assert meta.hires_image_url == AnyUrl(
        "https://www.lego.com/cdn/cs/set/assets/blt/75419_Prod.png?width=1500"
    )
    assert len(meta.images) == 1
    assert meta.images[0].url == AnyUrl("https://www.lego.com/asset1.png")
    assert meta.images[0].alt_text == "Front View"
    assert len(meta.videos) == 1
    assert meta.videos[0].title == "Designer Video"
    assert meta.videos[0].url == AnyUrl("https://www.lego.com/video.mp4")
    assert meta.videos[0].quality == "Highest"
    assert len(meta.pdfs) == 2
    assert meta.pdfs[0].filesize == 92919030
    assert meta.pdfs[0].sequence_number == 1
    assert meta.pdfs[0].sequence_total == 2
    assert meta.pdfs[0].preview_alt == "Booklet 1"


def test_parse_metadata_from_graphql_not_found():
    data = {
        "customerService": {
            "getBuildingInstructionsForSet": {
                "status": "Not Found",
                "data": {"name": ""},
            }
        },
        "product": None,
    }
    meta = parse_metadata_from_graphql(data, set_number="99999", locale="en-us")
    assert meta is None
