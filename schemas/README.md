# OpenAPI Schema and Model Generation

This directory contains the OpenAPI schema definition and tools for generating Pydantic models.

## Overview

- **`openapi.yaml`**: The source of truth for all data models shared between Python and other applications
- **`generate_models.sh`**: Script to generate Pydantic models from the OpenAPI schema

## Usage

### Generating Python Models

To regenerate Python models from the OpenAPI schema:

```bash
pants run src/build_a_long/schemas:generate_models
```

This will:

1. Read `schemas/openapi.yaml`
2. Generate Pydantic v2 models in `src/build_a_long/schemas/generated_models.py`
3. Automatically format the generated file with `pants fix`

### Updating the Schema

1. Edit `schemas/openapi.yaml` to add or modify data models
2. Run the generation command above to update Python models
3. Commit both the schema and generated models

## Schema Guidelines

- Use OpenAPI 3.1.0 format
- Include detailed descriptions for all types and fields
- Mark required fields explicitly
- Use appropriate data types (integer, string, boolean, etc.)
- Use `nullable: true` for optional fields that can be null
- Define reusable schemas under `components/schemas`

## Generated Models

The generated models are created using `datamodel-codegen` and should **not be edited manually**. Any changes should be made to `openapi.yaml` and regenerated.

Models include:

- Type hints for all fields
- Default values where specified
- Validation based on schema constraints
- Pydantic v2 BaseModel classes
