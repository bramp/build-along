# Configuration Standards

This directory contains configuration classes for the classifier components. To ensure consistency and usability, all configuration files must adhere to the following standards.

## 1. File and Class Naming

*   **Files**: Must use `snake_case` and end with `_config.py` (e.g., `arrow_config.py`).
*   **Classes**: Must use `PascalCase`, end with `Config`, and match the file name (e.g., `ArrowConfig` in `arrow_config.py`).

## 2. Structure and Types

*   **Base Class**: All configuration classes must inherit from `pydantic.BaseModel`.
*   **Imports**: consistently include `from __future__ import annotations`.
*   **Type Hints**: Use specific types where possible.
    *   `build_a_long.pdf_extract.classifier.score.Weight` for probabilities, weights, or score thresholds (0.0 - 1.0).
    *   `float` for dimensions (points) and ratios.
    *   `int` for counts.
    *   `bool` for flags.

## 3. Fields and Descriptions

*   **Pydantic Fields**: Use `pydantic.Field` with the `description` parameter. Do **not** use docstrings for fields.
*   **Defaults**: Provide sensible default values.

## 4. Naming Conventions & Existing Fields

This section lists observed field names and proposes standardizations.

### Scores & Weights (Type: `Weight`)

*   `min_score` (Standard across all)
*   `text_weight`
*   `font_size_weight`
*   `position_weight`
*   `size_weight`
*   `shape_weight` (Standardize: `box_shape_weight` -> `shape_weight`?)
*   `fill_color_weight`
*   `diagram_weight`
*   `count_weight`
*   `aspect_weight`
*   `proximity_weight`

### Dimensions & Sizes (Type: `float` in points)

*   `min_width`, `max_width`
*   `min_height`, `max_height`
*   `min_size`, `max_size` (General size/diameter)
*   `ideal_size`
*   `max_thickness`
*   `proximity_close_distance`, `proximity_far_distance`
*   **Renaming Proposals:**
    *   `icon_min_size` -> `min_icon_size`
    *   `min_part_width` -> `min_subassembly_part_width` (or similar context)
    *   `indicator_search_margin` -> `search_margin`?
    *   `overlap_expansion` -> `expansion_margin`?

### Ratios (Type: `float` 0.0-1.0)

*   `min_aspect_ratio`, `max_aspect_ratio` (Standardize `min_aspect` -> `min_aspect_ratio`)
*   `min_coverage_ratio`
*   `min_length_ratio`
*   `max_area_ratio`
*   `max_page_width_ratio`, `max_page_height_ratio`
*   **Renaming Proposals:**
    *   `icon_max_x_ratio` -> `max_icon_x_ratio`
    *   `bottom_margin_threshold` -> `max_bottom_margin_ratio` (if it's a ratio)
    *   `page_number_proximity_threshold` -> `max_page_number_proximity_ratio`

### Counts (Type: `int`)

*   **Renaming Proposals:**
    *   `min_drawings_in_cluster` -> `min_cluster_drawing_count`
    *   `max_drawings_in_cluster` -> `max_cluster_drawing_count`
    *   `min_text_blocks` -> `min_text_block_count`
    *   `min_total_characters` -> `min_character_count`

### Margins & Tolerances (Type: `float` in points)

*   `edge_margin`
*   `proximity_margin`
*   **Renaming Proposals:**
    *   `edge_tolerance` -> `edge_margin` (Consistency with DividerConfig)

### Multipliers & Scales (Type: `float`)

*   `position_scale`
*   `min_size_part_multiplier`
*   **Renaming Proposals:**
    *   `page_number_proximity_boost` -> `page_number_proximity_multiplier`

### Thresholds (Type: `float`)

*   `white_threshold`

## 5. Migration Checklist

- [ ] Convert `min_aspect` / `max_aspect` to `min_aspect_ratio` / `max_aspect_ratio`.
- [ ] Rename `edge_tolerance` to `edge_margin`.
- [ ] Ensure `weight` fields use `Weight` type.
- [ ] Move docstrings to `Field(description=...)`.
- [ ] Standardize `_count` suffix for integer counts.
