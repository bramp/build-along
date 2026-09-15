"""Debug script to understand shine vs progress_bar_indicator conflict."""

from pathlib import Path

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier import Classifier
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.extractor.extractor import ExtractionResult

# Load a failing test case
fixture_path = Path(
    "src/build_a_long/pdf_extract/fixtures/6509377_page_014_raw.json"
)

extraction = ExtractionResult.model_validate_json(fixture_path.read_text())
page = extraction.pages[0]

# Create classifier and just run scoring (no build phase)
config = ClassifierConfig()


class ScoringOnlyClassifier(Classifier):
    """Classifier that only runs scoring, not building."""

    def classify(self, page_data):
        result = ClassificationResult(page_data=page_data)
        for classifier in self.classifiers:
            classifier.score(result)
        return result


classifier = ScoringOnlyClassifier(config, use_solver_for=set())
result = classifier.classify(page)

# Get shine and progress_bar_indicator candidates
shine_candidates = list(result.get_scored_candidates("shine"))
pbi_candidates = list(result.get_scored_candidates("progress_bar_indicator"))
page_num_candidates = list(result.get_scored_candidates("page_number"))

print(f"=== Page {page.page_number} ===")
print(f"\nShine candidates: {len(shine_candidates)}")
for c in shine_candidates:
    print(f"  - {c.bbox} score={c.score:.3f}")
    print(f"    source_blocks: {[type(b).__name__ for b in c.source_blocks]}")
    for b in c.source_blocks:
        print(f"      {b.bbox}")

print(f"\nProgress bar indicator candidates: {len(pbi_candidates)}")
for c in pbi_candidates:
    print(f"  - {c.bbox} score={c.score:.3f}")
    print(f"    source_blocks: {[type(b).__name__ for b in c.source_blocks]}")
    for b in c.source_blocks:
        print(f"      {b.bbox}")

print(f"\nPage number candidates: {len(page_num_candidates)}")
for c in page_num_candidates:
    print(f"  - {c.bbox} score={c.score:.3f}")
    print(f"    source_blocks: {[type(b).__name__ for b in c.source_blocks]}")
    for b in c.source_blocks:
        print(f"      {b.bbox}")

# Check for block overlaps between shine and progress_bar_indicator
print("\n=== Block Overlap Analysis ===")
shine_blocks = set()
for c in shine_candidates:
    for b in c.source_blocks:
        shine_blocks.add(id(b))

pbi_blocks = set()
for c in pbi_candidates:
    for b in c.source_blocks:
        pbi_blocks.add(id(b))

page_num_blocks = set()
for c in page_num_candidates:
    for b in c.source_blocks:
        page_num_blocks.add(id(b))

shared_shine_pbi = shine_blocks & pbi_blocks
shared_shine_pagenum = shine_blocks & page_num_blocks
shared_pbi_pagenum = pbi_blocks & page_num_blocks

print(f"Shine blocks: {len(shine_blocks)}")
print(f"Progress bar indicator blocks: {len(pbi_blocks)}")
print(f"Page number blocks: {len(page_num_blocks)}")
print(f"Shared shine<->pbi: {len(shared_shine_pbi)}")
print(f"Shared shine<->page_num: {len(shared_shine_pagenum)}")
print(f"Shared pbi<->page_num: {len(shared_pbi_pagenum)}")

# Find which candidates share blocks
print("\n=== Candidates Sharing Blocks ===")
for shine_c in shine_candidates:
    shine_block_ids = {id(b) for b in shine_c.source_blocks}
    for pbi_c in pbi_candidates:
        pbi_block_ids = {id(b) for b in pbi_c.source_blocks}
        shared = shine_block_ids & pbi_block_ids
        if shared:
            print(f"CONFLICT: shine {shine_c.bbox} <-> pbi {pbi_c.bbox}")
            print(f"  Shared {len(shared)} blocks")

for shine_c in shine_candidates:
    shine_block_ids = {id(b) for b in shine_c.source_blocks}
    for pn_c in page_num_candidates:
        pn_block_ids = {id(b) for b in pn_c.source_blocks}
        shared = shine_block_ids & pn_block_ids
        if shared:
            print(f"CONFLICT: shine {shine_c.bbox} <-> page_num {pn_c.bbox}")
            print(f"  Shared {len(shared)} blocks")

# Now check with solver enabled for shine
print("\n=== With Shine in Solver ===")
labels_with_shine = {
    "parts_list",
    "part",
    "part_count",
    "part_image",
    "part_number",
    "piece_length",
    "shine",
    "page_number",
    "progress_bar",
    "progress_bar_bar",
    "progress_bar_indicator",
    "background",
    "divider",
    "step",
    "step_number",
}
classifier2 = Classifier(config, use_solver_for=labels_with_shine)
result2 = classifier2.classify(page)

# Check what was selected
selected_shine = [
    c for c in result2.get_scored_candidates("shine") if result2.is_selected(c)
]
selected_pbi = [
    c
    for c in result2.get_scored_candidates("progress_bar_indicator")
    if result2.is_selected(c)
]
selected_pn = [
    c
    for c in result2.get_scored_candidates("page_number")
    if result2.is_selected(c)
]

print(f"Selected shine: {len(selected_shine)}")
for c in selected_shine:
    print(f"  - {c.bbox} score={c.score:.3f}")

print(f"Selected pbi: {len(selected_pbi)}")
for c in selected_pbi:
    print(f"  - {c.bbox} score={c.score:.3f}")

print(f"Selected page_number: {len(selected_pn)}")
for c in selected_pn:
    print(f"  - {c.bbox} score={c.score:.3f}")

# Check if progress_bar was built
page_elem = result2.page
if page_elem:
    print(f"\nPage has progress_bar: {page_elem.progress_bar is not None}")
    print(f"Page has page_number: {page_elem.page_number is not None}")
