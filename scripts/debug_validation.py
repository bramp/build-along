#!/usr/bin/env python3
"""Debug script to investigate unconsumed blocks using the enhanced validation."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import ExtractionResult
from build_a_long.pdf_extract.fixtures import (
    FIXTURES_DIR,
    extract_element_id,
    load_classifier_config,
)
from build_a_long.pdf_extract.validation.rules import validate_unconsumed_blocks
from build_a_long.pdf_extract.validation import ValidationResult

# Load the raw page data
fixture_file = "6509377_page_072_raw.json"
fixture_path = FIXTURES_DIR / fixture_file
extraction = ExtractionResult.model_validate_json(fixture_path.read_text())
page_data = extraction.pages[0]

# Load classifier config with hints
element_id = extract_element_id(fixture_file)
config = load_classifier_config(element_id)

# Run classification
result = classify_elements(page_data, config)

# Focus on specific blocks  
print("="*60)
print("ANALYZING DUPLICATE TEXT BLOCKS AT (99.3,44.5,106.5,54.4)")
print("="*60)

# Find the text blocks at this location
target_bbox = "(99.3,44.5,106.5,54.4)"
text_blocks = [b for b in page_data.blocks if str(b.bbox) == target_bbox]
print(f"Found {len(text_blocks)} blocks with bbox {target_bbox}")

for block in text_blocks:
    print(f"\n  Block #{block.id} {type(block).__name__}: '{getattr(block, 'text', 'N/A')}'")

# Check what part_count candidates exist and their source_blocks
print("\n" + "="*60)
print("PART_COUNT CANDIDATES AND THEIR SOURCE BLOCKS")
print("="*60)

for cand in result.get_candidates("part_count"):
    # Only show candidates near this area
    if 90 < cand.bbox.x0 < 110 and 40 < cand.bbox.y0 < 60:
        selected = result.is_solver_selected(cand)
        built = cand.id in result._build_cache
        print(f"\npart_count id={cand.id} at {cand.bbox}")
        print(f"  selected={selected} built={built}")
        print(f"  source_blocks ({len(cand.source_blocks)}):")
        for sb in cand.source_blocks:
            print(f"    - #{sb.id} {type(sb).__name__} at {sb.bbox}")

# Check what Part candidates exist and their children
print("\n" + "="*60)
print("PART CANDIDATES WITH BBOX (98.8,22.2,139.9,54.4)")
print("="*60)

for cand in result.get_candidates("part"):
    if 98 < cand.bbox.x0 < 100 and 22 < cand.bbox.y0 < 23:
        selected = result.is_solver_selected(cand)
        built = cand.id in result._build_cache
        print(f"\npart id={cand.id} at {cand.bbox}")
        print(f"  selected={selected} built={built}")
        print(f"  source_blocks ({len(cand.source_blocks)}):")
        for sb in cand.source_blocks:
            print(f"    - #{sb.id} {type(sb).__name__} at {sb.bbox}")
        # Check score_details for child references
        if hasattr(cand.score_details, 'part_count'):
            pc = cand.score_details.part_count
            if pc:
                print(f"  part_count child: id={pc.id} at {pc.bbox}")

# Let's also check if there are duplicate Text blocks
print("\n" + "="*60)
print("ALL TEXT BLOCKS ON PAGE (checking for duplicates)")
print("="*60)

from collections import defaultdict
text_by_bbox = defaultdict(list)
for block in page_data.blocks:
    if hasattr(block, 'text'):
        text_by_bbox[str(block.bbox)].append(block)

for bbox_str, blocks in text_by_bbox.items():
    if len(blocks) > 1:
        print(f"\nDUPLICATE TEXT at {bbox_str}:")
        for b in blocks:
            print(f"  #{b.id} text='{b.text}'")

# Check why Drawing #57 wasn't included as an effect
print("\n" + "="*60)
print("INVESTIGATING DRAWING BLOCK #57")
print("="*60)

drawing_57 = [b for b in page_data.blocks if b.id == 57][0]
print(f"Drawing #57 bbox: {drawing_57.bbox}")

text_blocks_at_target = [b for b in page_data.blocks if str(b.bbox) == "(99.3,44.5,106.5,54.4)"]
for text_block in text_blocks_at_target[:1]:
    print(f"\nText block #{text_block.id} bbox: {text_block.bbox}")
    expanded = text_block.bbox.expand(2.0)
    print(f"  Expanded by 2.0: {expanded}")
    print(f"  Drawing #57 bbox: {drawing_57.bbox}")
    print(f"  Contains? {expanded.contains(drawing_57.bbox)}")
    
# Also check what additional_source_blocks would be found
from build_a_long.pdf_extract.classifier.block_filter import find_contained_effects

print("\n" + "="*60)
print("WHAT find_contained_effects RETURNS FOR TEXT BLOCK #1")
print("="*60)

text_block_1 = [b for b in page_data.blocks if b.id == 1][0]
effects = find_contained_effects(text_block_1, page_data.blocks, margin=2.0)
print(f"Effects found: {len(effects)}")
for e in effects:
    print(f"  - #{e.id} {type(e).__name__} at {e.bbox}")
    
# Simulate what PartCountClassifier._get_additional_source_blocks does
from build_a_long.pdf_extract.extractor.page_blocks import Drawing
drawing_effects = [b for b in effects if isinstance(b, Drawing)]
print(f"\nDrawing effects only: {len(drawing_effects)}")
for e in drawing_effects:
    print(f"  - #{e.id} {type(e).__name__} at {e.bbox}")

# Let's trace through what happens when part_count candidates are created
print("\n" + "="*60)
print("TRACING PART_COUNT CANDIDATE CREATION")
print("="*60)

# Get the classifier
from build_a_long.pdf_extract.classifier.classifier import Classifier
classifier = Classifier(config)

# Find PartCountClassifier
part_count_classifier = None
for c in classifier.classifiers:
    if c.output == "part_count":
        part_count_classifier = c
        break

print(f"PartCountClassifier found: {part_count_classifier}")
print(f"  effects_margin: {part_count_classifier.effects_margin}")

# Manually call _get_additional_source_blocks for block #1
from build_a_long.pdf_extract.classifier.classification_result import ClassificationResult
test_result = ClassificationResult(page_data=page_data)

# Debug: check what blocks are in test_result
print(f"\ntest_result.page_data.blocks count: {len(test_result.page_data.blocks)}")
print(f"  Drawing blocks: {[b for b in test_result.page_data.blocks if isinstance(b, Drawing)][:3]}")

# Debug the _get_additional_source_blocks method step-by-step
print(f"\nDirect call to method's logic:")
margin = part_count_classifier.effects_margin
print(f"  margin: {margin}")
if margin is not None:
    effects_inner = find_contained_effects(
        text_block_1,
        test_result.page_data.blocks,
        margin=margin,
    )
    print(f"  find_contained_effects returned: {len(effects_inner)}")
    for e in effects_inner:
        print(f"    - #{e.id} {type(e).__name__} isinstance(Drawing)={isinstance(e, Drawing)}")
    drawing_only = [b for b in effects_inner if isinstance(b, Drawing)]
    print(f"  Drawing filter: {len(drawing_only)}")

# Also trace what the method actually does - add debug logging to verify
import inspect
source = inspect.getsource(part_count_classifier._get_additional_source_blocks)
print(f"\nMethod source starts with: {source[:200]}...")

# Check which class the method is from
method = part_count_classifier._get_additional_source_blocks
print(f"\nMethod __qualname__: {method.__qualname__}")
print(f"Method owner: {method.__self__.__class__.__name__}")

# Trace into the method manually
print(f"\nTracing _get_additional_source_blocks manually:")
print(f"  self.effects_margin = {part_count_classifier.effects_margin}")
print(f"  result.page_data.blocks count = {len(test_result.page_data.blocks)}")

# Check if something is different about the Drawing import
from build_a_long.pdf_extract.extractor.page_blocks import Drawing as OurDrawing
from build_a_long.pdf_extract.classifier.parts.part_count_classifier import Drawing as ClassifierDrawing
print(f"  OurDrawing = {OurDrawing}")
print(f"  ClassifierDrawing = {ClassifierDrawing}")
print(f"  Are they same? {OurDrawing is ClassifierDrawing}")

# Try calling with print statements added
print(f"\nDirect call with debug:")
margin = part_count_classifier.effects_margin
print(f"  margin is not None: {margin is not None}")
if margin is not None:
    effects = find_contained_effects(
        text_block_1,
        test_result.page_data.blocks,
        margin=margin,
    )
    print(f"  effects count: {len(effects)}")
    drawings = [b for b in effects if isinstance(b, Drawing)]
    print(f"  drawings count: {len(drawings)}")
    for d in drawings:
        print(f"    Drawing #{d.id}")

# Check the method implementation more carefully - maybe it has different behavior 
# when called with different block types
print(f"\nDebug: Checking PartCountClassifier's method directly with instrumentation:")
def debug_get_additional_source_blocks(self, block, result):
    print(f"    Inside debug method")
    margin = self.effects_margin
    print(f"    margin = {margin}")
    if margin is not None:
        effects = find_contained_effects(
            block,
            result.page_data.blocks,
            margin=margin,
        )
        print(f"    effects = {len(effects)}")
        result_list = [b for b in effects if isinstance(b, Drawing)]
        print(f"    result_list = {len(result_list)}")
        return result_list
    return []

# Manually call the debug version
debug_result = debug_get_additional_source_blocks(part_count_classifier, text_block_1, test_result)
print(f"    debug_result: {len(debug_result)}")

additional = part_count_classifier._get_additional_source_blocks(text_block_1, test_result)
print(f"\n_get_additional_source_blocks for block #1:")
print(f"  Returns {len(additional)} blocks:")
for b in additional:
    print(f"    - #{b.id} {type(b).__name__} at {b.bbox}")
    
# Wait... maybe the issue is that the imported Drawing in part_count_classifier is different?
# Let me check what the actual method imports
print(f"\nChecking Drawing import inside the method:")
import build_a_long.pdf_extract.classifier.parts.part_count_classifier as pc_module
print(f"  pc_module.Drawing = {pc_module.Drawing}")
print(f"  Drawing #57 class = {type(drawing_57).__name__}")
print(f"  isinstance check: {isinstance(drawing_57, pc_module.Drawing)}")
