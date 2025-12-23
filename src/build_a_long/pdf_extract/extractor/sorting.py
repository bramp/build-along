"""Sorting utilities for LEGO page elements.

This module provides functions for sorting elements based on their bounding boxes
to produce deterministic, reading-order outputs.
"""

from __future__ import annotations

from collections.abc import Sequence

from build_a_long.pdf_extract.extractor.bbox import HasBBox


def sort_left_to_right[T: HasBBox](items: Sequence[T]) -> list[T]:
    """Sort items left-to-right by x0 position, with y0 as tiebreaker.

    Use this for elements that should appear in reading order when arranged
    horizontally (e.g., arrows, dividers, previews).

    Args:
        items: Sequence of items with bbox attributes.

    Returns:
        New list sorted by (x0, y0).
    """
    return sorted(items, key=lambda item: (item.bbox.x0, item.bbox.y0))


def sort_by_rows[T: HasBBox](items: Sequence[T]) -> list[T]:
    """Sort items row-by-row: top-to-bottom by row, left-to-right within each row.

    Items are grouped into rows based on vertical overlap. Items whose bounding
    boxes overlap vertically are considered to be in the same row.

    Use this for elements arranged in horizontal rows (e.g., parts in a PartsList).

    Args:
        items: Sequence of items with bbox attributes.

    Returns:
        New list sorted by rows (top to bottom), then left-to-right within rows.
    """
    if not items:
        return []

    # Sort by y-center first to process top-to-bottom
    sorted_by_y = sorted(items, key=lambda item: (item.bbox.y0 + item.bbox.y1) / 2)

    rows: list[list[T]] = []
    for item in sorted_by_y:
        # Try to find an existing row this item belongs to
        placed = False
        for row in rows:
            # Check if item overlaps vertically with any item in the row
            for row_item in row:
                row_y0, row_y1 = row_item.bbox.y0, row_item.bbox.y1
                item_y0, item_y1 = item.bbox.y0, item.bbox.y1
                # Check for vertical overlap
                if item_y0 < row_y1 and item_y1 > row_y0:
                    row.append(item)
                    placed = True
                    break
            if placed:
                break

        if not placed:
            rows.append([item])

    # Sort rows by average y-center, then sort items within each row by x0
    rows.sort(key=lambda row: sum((i.bbox.y0 + i.bbox.y1) / 2 for i in row) / len(row))
    result: list[T] = []
    for row in rows:
        row.sort(key=lambda item: item.bbox.x0)
        result.extend(row)

    return result


def sort_by_columns[T: HasBBox](items: Sequence[T]) -> list[T]:
    """Sort items column-by-column: left-to-right by column, top-to-bottom within.

    Items are grouped into columns based on horizontal overlap. Items whose
    bounding boxes overlap horizontally are considered to be in the same column.

    Use this for elements arranged in vertical columns (e.g., parts on catalog pages).

    Args:
        items: Sequence of items with bbox attributes.

    Returns:
        New list sorted by columns (left to right), then top-to-bottom within columns.
    """
    if not items:
        return []

    # Sort by x-center first to process left-to-right
    sorted_by_x = sorted(items, key=lambda item: (item.bbox.x0 + item.bbox.x1) / 2)

    columns: list[list[T]] = []
    for item in sorted_by_x:
        # Try to find an existing column this item belongs to
        placed = False
        for col in columns:
            # Check if item overlaps horizontally with any item in the column
            for col_item in col:
                col_x0, col_x1 = col_item.bbox.x0, col_item.bbox.x1
                item_x0, item_x1 = item.bbox.x0, item.bbox.x1
                # Check for horizontal overlap
                if item_x0 < col_x1 and item_x1 > col_x0:
                    col.append(item)
                    placed = True
                    break
            if placed:
                break

        if not placed:
            columns.append([item])

    # Sort columns by average x-center, then sort items within each column by y0
    columns.sort(
        key=lambda col: sum((i.bbox.x0 + i.bbox.x1) / 2 for i in col) / len(col)
    )
    result: list[T] = []
    for col in columns:
        col.sort(key=lambda item: item.bbox.y0)
        result.extend(col)

    return result
