#!/usr/bin/env python3
"""Analyze shine placement in golden files."""

import json
from pathlib import Path


def analyze_page(raw_path: Path):
    """Analyze a raw page file for images and shine-like drawings."""
    data = json.loads(raw_path.read_text())
    
    # Handle both formats: direct blocks or pages wrapper
    if "pages" in data:
        blocks = data["pages"][0].get("blocks", [])
    else:
        blocks = data.get("blocks", [])
    
    images = []
    shines = []
    
    for block in blocks:
        tag = block.get("__tag__")
        bbox = block["bbox"]
        
        if tag == "Image":
            images.append(bbox)
        elif tag == "Drawing":
            w = bbox["x1"] - bbox["x0"]
            h = bbox["y1"] - bbox["y0"]
            if 5 <= w <= 15 and 5 <= h <= 15:
                aspect = w / h if h > 0 else 0
                if 0.7 <= aspect <= 1.4:
                    shines.append(bbox)
    
    return images, shines


def main():
    fixtures_dir = Path("src/build_a_long/pdf_extract/fixtures")
    
    # Analyze specific pages with shine
    for page_file in ["6509377_page_176_raw.json", "6509377_page_014_raw.json"]:
        raw_path = fixtures_dir / page_file
        if not raw_path.exists():
            continue
            
        print(f"\n=== {page_file} ===")
        images, shines = analyze_page(raw_path)
        
        print("Images:")
        for bbox in images:
            print(f"  ({bbox['x0']:.1f},{bbox['y0']:.1f},{bbox['x1']:.1f},{bbox['y1']:.1f})")
        
        print("Shine-like drawings:")
        for bbox in shines:
            print(f"  ({bbox['x0']:.1f},{bbox['y0']:.1f},{bbox['x1']:.1f},{bbox['y1']:.1f})")
        
        # Calculate distances from image TR corner to shine center
        print("\nDistances (image TR corner to shine center):")
        for img in images:
            img_tr_x = img["x1"]
            img_tr_y = img["y0"]
            
            for shine in shines:
                sc_x = (shine["x0"] + shine["x1"]) / 2
                sc_y = (shine["y0"] + shine["y1"]) / 2
                dist = ((sc_x - img_tr_x)**2 + (sc_y - img_tr_y)**2)**0.5
                
                # Check if shine overlaps image
                overlaps = (
                    shine["x0"] < img["x1"] and shine["x1"] > img["x0"] and
                    shine["y0"] < img["y1"] and shine["y1"] > img["y0"]
                )
                
                if overlaps:
                    print(f"  Image ({img['x0']:.1f},{img['y0']:.1f},{img['x1']:.1f},{img['y1']:.1f})")
                    print(f"    -> Shine ({shine['x0']:.1f},{shine['y0']:.1f},{shine['x1']:.1f},{shine['y1']:.1f})")
                    print(f"    Distance: {dist:.1f}, overlaps: {overlaps}")


if __name__ == "__main__":
    main()
