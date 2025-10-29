"""Analyze which elements are being deleted and why."""

from pathlib import Path
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.classifier.classifier import classify_elements

fixture = Path('data/75375/6509377_page_014_raw.json')
page_data = PageData.from_json(fixture.read_text())
assert isinstance(page_data, PageData), "Expected single PageData, got list"
page: PageData = page_data

print(f"Page has {len(page.elements)} elements before classification")
print(f"Deleted elements before: {sum(1 for e in page.elements if e.deleted)}")

# Classify the page
result = classify_elements(page)

print(f"\nPage has {len(page.elements)} elements after classification")
labeled_count = sum(
    1 for e in page.elements if result.get_label(e) is not None)
print(f"Labeled elements after: {labeled_count}")
print(f"Deleted elements after: {sum(1 for e in page.elements if e.deleted)}")

# Find deleted labeled elements
deleted_labeled = [
    e for e in page.elements if e.deleted and result.get_label(e) is not None]

print(f'\n{"="*60}')
print(f'Found {len(deleted_labeled)} deleted labeled elements:')
print(f'{"="*60}')
for elem in deleted_labeled:
    text = getattr(elem, 'text', 'N/A')
    label = result.get_label(elem)
    print(f'  ID {elem.id}: {label} "{text}" at bbox {elem.bbox}')

# Find non-deleted with same label at same location
print(f'\n{"="*60}')
print('Checking for duplicates at same location:')
print(f'{"="*60}')
labeled = [e for e in page.elements if result.get_label(
    e) is not None and not e.deleted]
for del_elem in deleted_labeled:
    del_bbox = del_elem.bbox
    del_label = result.get_label(del_elem)
    del_text = getattr(del_elem, 'text', 'N/A')

    # Check for exact match
    exact = [e for e in labeled if result.get_label(
        e) == del_label and e.bbox == del_bbox]

    # Check for near match (similar bbox)
    near = []
    for e in labeled:
        if result.get_label(e) == del_label:
            iou = del_bbox.iou(e.bbox)
            if iou > 0.7:
                near.append((e, iou))

    if exact:
        print(
            f'  ✓ ID {del_elem.id} ({del_label} "{del_text}") has {len(exact)} EXACT duplicates: IDs {[e.id for e in exact]}')
        for e in exact:
            e_text = getattr(e, 'text', 'N/A')
            print(f'      → ID {e.id}: "{e_text}"')
    elif near:
        print(
            f'  ~ ID {del_elem.id} ({del_label} "{del_text}") has {len(near)} SIMILAR elements (IOU > 0.7):')
        for e, iou in near:
            e_text = getattr(e, 'text', 'N/A')
            print(f'      → ID {e.id}: "{e_text}" at {e.bbox} (IOU={iou:.2f})')
    else:
        print(
            f'  ✗ ID {del_elem.id} ({del_label} "{del_text}") has NO duplicate - THIS IS THE BUG!')
        # Let's see what's nearby
        print(f'      Looking for other {del_label} elements:')
        same_label = [e for e in labeled if result.get_label(e) == del_label]
        for e in same_label[:3]:  # Show first 3
            e_text = getattr(e, 'text', 'N/A')
            iou = del_bbox.iou(e.bbox)
            print(
                f'        → ID {e.id}: "{e_text}" at {e.bbox} (IOU={iou:.2f})')
