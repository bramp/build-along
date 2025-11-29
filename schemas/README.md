# JSON Schema Files

This directory contains JSON Schema files (in YAML format for readability) that describe the data structures used by this project. These schemas are **generated from Pydantic models** in Python, making Python the source of truth.

## Schema Files

- **`lego_manual.schema.yaml`**: Schema for parsed LEGO instruction manuals (pages, steps, parts lists, etc.)
- **`lego_metadata.schema.yaml`**: Schema for LEGO set metadata (set info, PDF URLs, etc.)

## Generating Schemas

The schemas are generated from Pydantic models using:

```bash
# Generate manual schema (from lego_page_elements.py)
pants run src/build_a_long/schemas:generate_schema -- manual schemas/lego_manual.schema.yaml

# Generate metadata schema (from downloader/models.py)
pants run src/build_a_long/schemas:generate_schema -- metadata schemas/lego_metadata.schema.yaml
```

## Source of Truth

- **Manual schema**: `src/build_a_long/pdf_extract/extractor/lego_page_elements.py`
- **Metadata schema**: `src/build_a_long/downloader/models.py`

## Usage

These schemas can be used by other applications (in any language) to:

1. **Validate** JSON data against the schema
2. **Generate** type definitions or models for other languages
3. **Document** the data structures

### Example: TypeScript

```bash
# Using json-schema-to-typescript
npx json-schema-to-typescript schemas/lego_manual.schema.yaml > lego_manual.d.ts
```

### Example: Validation

```python
import jsonschema
import yaml

with open("schemas/lego_manual.schema.yaml") as f:
    schema = yaml.safe_load(f)

# Validate your data
jsonschema.validate(your_data, schema)
```
