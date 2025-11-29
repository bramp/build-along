#!/usr/bin/env python3
"""Generate JSON Schema from Pydantic models.

This script generates JSON Schema files from Pydantic models. The generated
schemas can be used by other applications (in any language) to validate and
work with LEGO instruction data.

Usage:
    # Generate manual schema (parsed PDF structure)
    pants run src/build_a_long/schemas:generate_schema -- manual \
        schemas/lego_manual.schema.yaml

    # Generate metadata schema (downloaded set metadata)
    pants run src/build_a_long/schemas:generate_schema -- metadata \
        schemas/lego_metadata.schema.yaml

The schema is written to the specified output path. Supports both .json and .yaml
extensions (YAML is recommended for readability).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from build_a_long.downloader.models import InstructionMetadata
from build_a_long.pdf_extract.extractor.lego_page_elements import Manual


def generate_manual_schema() -> dict[str, Any]:
    """Generate JSON Schema for LEGO instruction manual models.

    Returns:
        A JSON Schema dict with all model definitions for parsed PDFs.
    """
    schema = Manual.model_json_schema(
        mode="serialization",
        by_alias=True,
    )

    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "LEGO Instruction Manual Schema"
    schema["description"] = (
        "JSON Schema for LEGO instruction manuals, generated from Pydantic models. "
        "This schema describes the structure of parsed LEGO instruction PDFs, "
        "including pages, steps, parts lists, and all visual elements."
    )

    return schema


def generate_metadata_schema() -> dict[str, Any]:
    """Generate JSON Schema for LEGO set metadata models.

    Returns:
        A JSON Schema dict with all model definitions for set metadata.
    """
    schema = InstructionMetadata.model_json_schema(
        mode="serialization",
    )

    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "LEGO Set Metadata Schema"
    schema["description"] = (
        "JSON Schema for LEGO set metadata, generated from Pydantic models. "
        "This schema describes the structure of metadata.json files containing "
        "information about LEGO sets and their instruction PDFs."
    )

    return schema


def write_schema(schema: dict[str, Any], output_path: Path) -> None:
    """Write schema to file in JSON or YAML format based on extension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        if output_path.suffix in (".yaml", ".yml"):
            yaml.dump(
                schema,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=88,
            )
        else:
            json.dump(schema, f, indent=2)
            f.write("\n")


def main() -> int:
    """Generate and write the JSON Schema file."""
    parser = argparse.ArgumentParser(
        description="Generate JSON Schema from Pydantic LEGO models."
    )
    parser.add_argument(
        "schema_type",
        choices=["manual", "metadata"],
        help="Which schema to generate: 'manual' for parsed PDFs, "
        "'metadata' for set metadata",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output path for the schema file (.json or .yaml)",
    )
    args = parser.parse_args()

    # Generate the requested schema
    if args.schema_type == "manual":
        schema = generate_manual_schema()
    else:
        schema = generate_metadata_schema()

    # Write to file
    write_schema(schema, args.output)

    print(f"Generated: {args.output}")

    # Print stats
    definitions = schema.get("$defs", {})
    print(f"  - {len(definitions)} model definitions")
    if definitions:
        print(f"  - Models: {', '.join(sorted(definitions.keys()))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
