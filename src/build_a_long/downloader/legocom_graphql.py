"""LEGO.com GraphQL API client for instructions and product metadata."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any
from urllib.parse import urljoin

import httpx
from pydantic import AnyUrl

from build_a_long.downloader.models import (
    Dimensions,
    ImageEntry,
    InstructionMetadata,
    PdfEntry,
    VideoEntry,
)
from build_a_long.downloader.util import clean_features_text

LEGO_BASE = "https://www.lego.com"

log = logging.getLogger(__name__)


def format_locale_for_header(locale: str) -> str:
    """Format a locale string like 'en-us' to 'en-US' for the x-locale header."""
    parts = locale.split("-")
    if len(parts) == 2:
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return locale


def get_graphql_auth_token(
    client: httpx.Client,
    locale: str = "en-us",
    base: str = LEGO_BASE,
) -> str | None:
    """Fetch an anonymous authentication token for LEGO.com GraphQL API.

    Sets the 'gqauth' cookie and 'Authorization' header on the client if successful.
    """
    for cookie_name in client.cookies:
        if cookie_name == "gqauth":
            return client.cookies["gqauth"]

    login_url = f"{base}/api/graphql/Login"
    headers = {
        "x-locale": format_locale_for_header(locale),
    }
    try:
        response = client.post(
            login_url,
            json={
                "operationName": "Login",
                "query": "mutation Login { login }",
                "variables": {},
            },
            headers=headers,
        )
        if response.status_code == 200:
            data = response.json()
            token = data.get("data", {}).get("login")
            if token and isinstance(token, str):
                client.cookies.set("gqauth", token, domain=".lego.com")
                client.headers["Authorization"] = token
                return token
    except Exception as e:
        log.debug("Failed to obtain LEGO GraphQL auth token: %s", e)

    return None


def fetch_set_data_graphql(
    client: httpx.Client,
    set_number: str,
    locale: str = "en-us",
    base: str = LEGO_BASE,
) -> dict[str, Any] | None:
    """Fetch both instruction data and product catalog data via GraphQL in a single request.

    Queries both `customerService.getBuildingInstructionsForSet` and `product`.

    Args:
        client: The HTTP client.
        set_number: The LEGO set number.
        locale: The locale to use.
        base: The base URL for the LEGO.com API.

    Returns:
        A dict containing 'customerService' and optional 'product' dicts,
        or None on network/API failure.
    """
    token = get_graphql_auth_token(client, locale=locale, base=base)

    query = f"""query {{
  customerService {{
    getBuildingInstructionsForSet(setNumber: "{set_number}") {{
      status
      data {{
        setNumber
        name
        year
        ageRating
        setPieceCount
        theme {{
          themeName
        }}
        setImage {{
          src
          alt
        }}
        buildingInstructions {{
          isAdditionalInfoBooklet
          sequence {{
            element
            total
          }}
          pdf {{
            pdfUrl
            fileSize
            coverImage {{
              src
              alt
            }}
          }}
        }}
      }}
    }}
  }}
  product(productCode: "{set_number}") {{
    id
    productCode
    name
    slug
    metaTitle
    metaDescription
    description
    featuresText
    primaryImage
    hires: primaryImage(size: HIRES)
    thumbnail: primaryImage(size: THUMBNAIL)
    productCategories {{
      id
      name
      slug
    }}
    brandCategory {{
      id
      name
      slug
    }}
    programCategory {{
      id
      name
      slug
    }}
    productMediaAssets {{
      __typename
      ... on ProductAssetImage {{
        id
        url
        altText
      }}
      ... on ProductAssetVideo {{
        id
        video {{
          title
          description
          videoFormats {{
            url
            quality
          }}
        }}
      }}
    }}
    ... on SingleVariantProduct {{
      variant {{
        id
        sku
        price {{
          centAmount
          formattedAmount
          currencyCode
        }}
        attributes {{
          minifigureCount
          buildHeight
          buildWidth
          buildDepth
          availabilityStatus
          availabilityText
          rating
          featuredFlags {{
            key
            label
          }}
        }}
      }}
    }}
  }}
}}"""

    headers = {
        "x-locale": format_locale_for_header(locale),
    }
    if token:
        headers["Authorization"] = token

    url = f"{base}/api/graphql/ProductDetails"
    try:
        response = client.post(url, json={"query": query}, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if "data" in result and isinstance(result["data"], dict):
                return result["data"]
            if "errors" in result:
                log.debug(
                    "GraphQL query returned errors for set %s: %s",
                    set_number,
                    result["errors"],
                )
    except Exception as e:
        log.debug("Error executing GraphQL query for set %s: %s", set_number, e)

    return None


def parse_metadata_from_graphql(
    data: dict[str, Any],
    set_number: str,
    locale: str,
    base: str = LEGO_BASE,
) -> InstructionMetadata | None:
    """Parse GraphQL response dictionary into an InstructionMetadata model.

    Returns None if the set was not found.
    """
    cs_obj = data.get("customerService", {}).get("getBuildingInstructionsForSet", {})
    cs_status = cs_obj.get("status")
    cs_data = cs_obj.get("data") or {}

    # If status is "Not Found" or data has no name, check product data as well
    product_data = data.get("product") or {}
    name = cs_data.get("name") or product_data.get("name")

    if not name and cs_status == "Not Found":
        return None

    # Core instruction metadata
    theme = None
    if cs_data.get("theme"):
        theme = cs_data["theme"].get("themeName")
    if not theme and product_data.get("brandCategory"):
        theme = product_data["brandCategory"].get("name")

    age = cs_data.get("ageRating")
    pieces = None
    if cs_data.get("setPieceCount"):
        with suppress(ValueError, TypeError):
            pieces = int(cs_data["setPieceCount"])

    year = None
    if cs_data.get("year"):
        with suppress(ValueError, TypeError):
            year = int(cs_data["year"])

    set_image_url = None
    set_image_alt = None
    if cs_data.get("setImage"):
        src = cs_data["setImage"].get("src")
        if src:
            set_image_url = urljoin(base, src)
        set_image_alt = cs_data["setImage"].get("alt")
    elif product_data.get("primaryImage"):
        set_image_url = product_data["primaryImage"]

    # Product catalog fields
    description = product_data.get("description")
    features_text = clean_features_text(product_data.get("featuresText"), description)
    meta_description = product_data.get("metaDescription")
    meta_title = product_data.get("metaTitle")
    slug = product_data.get("slug")

    hires_image_url = None
    if product_data.get("hires"):
        with suppress(Exception):
            hires_image_url = AnyUrl(product_data["hires"])

    thumbnail_image_url = None
    if product_data.get("thumbnail"):
        with suppress(Exception):
            thumbnail_image_url = AnyUrl(product_data["thumbnail"])

    # Gallery images and videos
    images: list[ImageEntry] = []
    videos: list[VideoEntry] = []
    for asset in product_data.get("productMediaAssets") or []:
        if not isinstance(asset, dict):
            continue
        typename = asset.get("__typename")
        if typename == "ProductAssetImage":
            url_str = asset.get("url")
            if url_str:
                with suppress(Exception):
                    images.append(
                        ImageEntry(
                            id=asset.get("id"),
                            url=AnyUrl(url_str),
                            alt_text=asset.get("altText") or None,
                        )
                    )
        elif typename == "ProductAssetVideo":
            video_obj = asset.get("video") or {}
            video_url = None
            quality = None
            formats = video_obj.get("videoFormats") or []
            if formats and isinstance(formats, list) and isinstance(formats[0], dict):
                first_format = formats[0]
                first_format_url = first_format.get("url")
                if first_format_url:
                    with suppress(Exception):
                        video_url = AnyUrl(first_format_url)
                quality = first_format.get("quality") or None
            videos.append(
                VideoEntry(
                    id=asset.get("id"),
                    title=video_obj.get("title") or None,
                    description=video_obj.get("description") or None,
                    url=video_url,
                    quality=quality,
                )
            )

    # Categories
    categories: list[str] = []
    for cat in product_data.get("productCategories") or []:
        if isinstance(cat, dict) and cat.get("name"):
            cat_name = cat["name"].strip()
            if cat_name and cat_name not in categories:
                categories.append(cat_name)

    # Brand
    brand = None
    brand_cat = product_data.get("brandCategory")
    if isinstance(brand_cat, dict) and brand_cat.get("name"):
        brand = brand_cat["name"]

    # Product variant attributes: minifigures, dimensions, availability, price
    minifigure_count = None
    dimensions = None
    availability_status = None
    availability_text = None
    price_formatted = None
    price_cents = None
    currency = None
    rating = None
    sku = None
    flags: list[str] = []

    variant = product_data.get("variant")
    if isinstance(variant, dict):
        sku = variant.get("sku")
        price_obj = variant.get("price")
        if isinstance(price_obj, dict):
            price_formatted = price_obj.get("formattedAmount")
            price_cents = price_obj.get("centAmount")
            currency = price_obj.get("currencyCode")

        attrs = variant.get("attributes")
        if isinstance(attrs, dict):
            if attrs.get("minifigureCount") is not None:
                with suppress(ValueError, TypeError):
                    minifigure_count = int(attrs["minifigureCount"])
            h = attrs.get("buildHeight")
            w = attrs.get("buildWidth")
            d = attrs.get("buildDepth")
            if h is not None or w is not None or d is not None:
                dimensions = Dimensions(
                    height=float(h) if h is not None else None,
                    width=float(w) if w is not None else None,
                    depth=float(d) if d is not None else None,
                )
            availability_status = attrs.get("availabilityStatus")
            availability_text = attrs.get("availabilityText")
            if attrs.get("rating") is not None:
                with suppress(ValueError, TypeError):
                    rating = float(attrs["rating"])
            for flag in attrs.get("featuredFlags") or []:
                if isinstance(flag, dict) and flag.get("label"):
                    flag_label = flag["label"].strip()
                    if flag_label and flag_label not in flags:
                        flags.append(flag_label)

    # PDFs
    pdfs: list[PdfEntry] = []
    for item in cs_data.get("buildingInstructions") or []:
        if not isinstance(item, dict):
            continue
        pdf_dict = item.get("pdf") or {}
        pdf_url_str = pdf_dict.get("pdfUrl")
        if not pdf_url_str:
            continue

        url = AnyUrl(urljoin(base, pdf_url_str))

        filesize = None
        if "fileSize" in pdf_dict:
            with suppress(ValueError, TypeError):
                filesize = int(pdf_dict["fileSize"])

        cover_image = pdf_dict.get("coverImage") or {}
        preview_url = None
        preview_alt = cover_image.get("alt") or None
        if cover_image.get("src"):
            preview_url = AnyUrl(urljoin(base, cover_image["src"]))

        seq_dict = item.get("sequence") or {}
        seq_num = None
        seq_tot = None
        if "element" in seq_dict:
            with suppress(ValueError, TypeError):
                seq_num = int(seq_dict["element"])
        if "total" in seq_dict:
            with suppress(ValueError, TypeError):
                seq_tot = int(seq_dict["total"])

        is_additional = None
        if "isAdditionalInfoBooklet" in item:
            with suppress(ValueError, TypeError):
                is_additional = bool(item["isAdditionalInfoBooklet"])

        pdfs.append(
            PdfEntry(
                url=url,
                filename=None,
                filesize=filesize,
                preview_url=preview_url,
                preview_alt=preview_alt,
                sequence_number=seq_num,
                sequence_total=seq_tot,
                is_additional_info_booklet=is_additional,
            )
        )

    return InstructionMetadata(
        set=set_number,
        locale=locale,
        name=name,
        theme=theme,
        age=age,
        pieces=pieces,
        year=year,
        set_image_url=AnyUrl(set_image_url) if set_image_url else None,
        set_image_alt=set_image_alt,
        description=description,
        features_text=features_text,
        meta_description=meta_description,
        meta_title=meta_title,
        slug=slug,
        hires_image_url=hires_image_url,
        thumbnail_image_url=thumbnail_image_url,
        images=images,
        videos=videos,
        categories=categories,
        brand=brand,
        minifigure_count=minifigure_count,
        dimensions=dimensions,
        availability_status=availability_status,
        availability_text=availability_text,
        price_formatted=price_formatted,
        price_cents=price_cents,
        currency=currency,
        rating=rating,
        sku=sku,
        flags=flags,
        pdfs=pdfs,
    )
