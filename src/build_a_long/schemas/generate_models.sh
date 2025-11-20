#!/usr/bin/env bash
# Generate Pydantic models from OpenAPI schema

set -euo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1

datamodel-codegen \
    --input schemas/openapi.yaml \
    --output src/build_a_long/schemas/generated_models.py \
    --input-file-type openapi \
    --use-default-kwarg \
    --use-default \
    --field-constraints \
    --output-model-type pydantic_v2.BaseModel \
    --use-union-operator \
    --strict-nullable \
    --custom-file-header "# Generated from schemas/openapi.yaml - DO NOT EDIT MANUALLY
# Use 'pants run src/build_a_long/schemas:generate_models' to regenerate this file"

# Ensure this generated file is formatted
PANTS_CONCURRENT=True pants fix src/build_a_long/schemas/generated_models.py

echo "✓ Generated models updated successfully"

