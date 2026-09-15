#!/usr/bin/env python3
"""Debug step number font sizes."""
import json
from pathlib import Path

fixture_path = Path("src/build_a_long/pdf_extract/fixtures/6433200_page_005_raw.json")
with open(fixture_path) as f:
    d = json.load(f)

blocks = d["pages"][0]["blocks"]
texts = [b for b in blocks if b.get("__type__") == "Text"]
for t in texts[:20]:
    h = t["bbox"]["y1"] - t["bbox"]["y0"]
    print(f"id={t['id']}, font_size={t.get('font_size')}, bbox_h={h:.1f}, text={t['text'][:30]!r}")
