#!/usr/bin/env python3
"""Analyze distances between part images and part counts in golden files."""

import json
from pathlib import Path

fixtures_dir = Path("src/build_a_long/pdf_extract/fixtures")
distances: list[tuple[float, float, str]] = []

for f in fixtures_dir.glob("*_expected.json"):
    data = json.loads(f.read_text())

    def find_parts(obj: object) -> None:
        if isinstance(obj, dict):
            if obj.get("__tag__") == "Part":
                count = obj.get("count", {})
                diagram = obj.get("diagram", {})
                if count and diagram:
                    count_bbox = count.get("bbox", {})
                    diagram_bbox = diagram.get("bbox", {})
                    if count_bbox and diagram_bbox:
                        distance = count_bbox.get("y0", 0) - diagram_bbox.get("y1", 0)
                        align = abs(count_bbox.get("x0", 0) - diagram_bbox.get("x0", 0))
                        distances.append((distance, align, f.name))
            for v in obj.values():
                find_parts(v)
        elif isinstance(obj, list):
            for item in obj:
                find_parts(item)

    find_parts(data)

distances.sort()
print(f"Total parts found: {len(distances)}")
print(f"\nDistance statistics (image bottom to count top):")
print(f"  Min: {min(d[0] for d in distances):.2f}")
print(f"  Max: {max(d[0] for d in distances):.2f}")
print(f"  Median: {sorted(d[0] for d in distances)[len(distances)//2]:.2f}")

print(f"\nDistance distribution:")
for threshold in [0, 1, 2, 5, 10, 20, 50]:
    count = sum(1 for d in distances if d[0] <= threshold)
    print(f"  <= {threshold}: {count} ({100*count/len(distances):.1f}%)")

print(f"\nAlignment offset statistics (x0 difference):")
print(f"  Min: {min(d[1] for d in distances):.2f}")
print(f"  Max: {max(d[1] for d in distances):.2f}")
print(f"  Median: {sorted(d[1] for d in distances)[len(distances)//2]:.2f}")

print(f"\nExamples with distance > 5:")
for d, a, f in distances:
    if d > 5:
        print(f"  dist={d:.2f}, align={a:.2f} in {f}")
