"""
Tool to analyze classifier performance across all PDFs in the data directory.

This script scans all PDF files, runs the classifier, and generates a report
showing how well the classifier performs (e.g., which pages have/don't have
page numbers identified).
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pymupdf

from build_a_long.bounding_box_extractor.classifier import classify_elements
from build_a_long.bounding_box_extractor.extractor import (
    PageData,
    extract_bounding_boxes,
)
from build_a_long.bounding_box_extractor.extractor.page_elements import Text

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PageAnalysis:
    """Analysis results for a single page."""

    page_number: int
    has_page_number_label: bool
    page_number_text: str | None
    page_number_score: float | None
    total_elements: int
    text_elements: int


@dataclass
class DocumentAnalysis:
    """Analysis results for a single PDF document."""

    pdf_path: Path
    total_pages: int
    pages_with_page_number: int
    pages: List[PageAnalysis]

    @property
    def page_number_coverage(self) -> float:
        """Percentage of pages with identified page numbers."""
        if self.total_pages == 0:
            return 0.0
        return (self.pages_with_page_number / self.total_pages) * 100


@dataclass
class GlobalAnalysis:
    """Analysis results across all documents."""

    documents: List[DocumentAnalysis]
    total_pdfs: int
    total_pages: int
    total_pages_with_page_number: int

    @property
    def overall_coverage(self) -> float:
        """Overall percentage of pages with identified page numbers."""
        if self.total_pages == 0:
            return 0.0
        return (self.total_pages_with_page_number / self.total_pages) * 100


def analyze_page(page_data: PageData) -> PageAnalysis:
    """Analyze classification results for a single page.

    Args:
        page_data: The page data with classified elements

    Returns:
        Analysis results for the page
    """
    total_elements = len(page_data.elements)
    text_elements = sum(1 for e in page_data.elements if isinstance(e, Text))

    # Find page number element
    page_number_element = None
    for element in page_data.elements:
        if isinstance(element, Text) and element.label == "page_number":
            page_number_element = element
            break

    has_page_number = page_number_element is not None
    page_number_text = page_number_element.text if page_number_element else None
    page_number_score = (
        page_number_element.label_scores.get("page_number")
        if page_number_element
        else None
    )

    return PageAnalysis(
        page_number=page_data.page_number,
        has_page_number_label=has_page_number,
        page_number_text=page_number_text,
        page_number_score=page_number_score,
        total_elements=total_elements,
        text_elements=text_elements,
    )


def analyze_document(pdf_path: Path) -> DocumentAnalysis:
    """Analyze classification results for a single PDF document.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Analysis results for the document
    """
    logger.info(f"Analyzing {pdf_path}")

    with pymupdf.open(str(pdf_path)) as doc:
        # Extract and classify all pages
        pages_data = extract_bounding_boxes(doc, start_page=None, end_page=None)
        classify_elements(pages_data)

        # Analyze each page
        page_analyses = [analyze_page(page_data) for page_data in pages_data]

        pages_with_page_number = sum(
            1 for pa in page_analyses if pa.has_page_number_label
        )

        return DocumentAnalysis(
            pdf_path=pdf_path,
            total_pages=len(pages_data),
            pages_with_page_number=pages_with_page_number,
            pages=page_analyses,
        )


def find_pdfs(data_dir: Path) -> List[Path]:
    """Find all PDF files in the data directory.

    Args:
        data_dir: Root data directory to search

    Returns:
        List of PDF file paths
    """
    return sorted(data_dir.rglob("*.pdf"))


def print_summary_report(analysis: GlobalAnalysis) -> None:
    """Print a summary report of the analysis.

    Args:
        analysis: Global analysis results
    """
    print("\n" + "=" * 80)
    print("CLASSIFIER ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal PDFs analyzed: {analysis.total_pdfs}")
    print(f"Total pages analyzed: {analysis.total_pages}")
    print(f"Pages with page number identified: {analysis.total_pages_with_page_number}")
    print(f"Overall coverage: {analysis.overall_coverage:.1f}%")

    # Group by coverage percentage
    coverage_buckets: Dict[str, List[DocumentAnalysis]] = {
        "100%": [],
        "75-99%": [],
        "50-74%": [],
        "25-49%": [],
        "1-24%": [],
        "0%": [],
    }

    for doc in analysis.documents:
        coverage = doc.page_number_coverage
        if coverage == 100.0:
            coverage_buckets["100%"].append(doc)
        elif coverage >= 75.0:
            coverage_buckets["75-99%"].append(doc)
        elif coverage >= 50.0:
            coverage_buckets["50-74%"].append(doc)
        elif coverage >= 25.0:
            coverage_buckets["25-49%"].append(doc)
        elif coverage > 0.0:
            coverage_buckets["1-24%"].append(doc)
        else:
            coverage_buckets["0%"].append(doc)

    print("\n" + "-" * 80)
    print("COVERAGE DISTRIBUTION")
    print("-" * 80)
    for bucket_name, docs in coverage_buckets.items():
        if docs:
            print(f"\n{bucket_name} coverage ({len(docs)} PDFs):")
            for doc in docs[:5]:  # Show first 5 in each bucket
                print(
                    f"  {doc.pdf_path.parent.name}/{doc.pdf_path.name}: "
                    f"{doc.pages_with_page_number}/{doc.total_pages} pages"
                )
            if len(docs) > 5:
                print(f"  ... and {len(docs) - 5} more")


def print_detailed_report(analysis: GlobalAnalysis, max_docs: int = 10) -> None:
    """Print a detailed report showing per-document and per-page results.

    Args:
        analysis: Global analysis results
        max_docs: Maximum number of documents to show in detail
    """
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    # Show documents with issues (not 100% coverage)
    docs_with_issues = [
        doc for doc in analysis.documents if doc.page_number_coverage < 100.0
    ]

    if docs_with_issues:
        print(f"\nDocuments with missing page numbers ({len(docs_with_issues)}):")
        for doc in docs_with_issues[:max_docs]:
            print(f"\n{doc.pdf_path.parent.name}/{doc.pdf_path.name}:")
            print(
                f"  Coverage: {doc.pages_with_page_number}/{doc.total_pages} "
                f"({doc.page_number_coverage:.1f}%)"
            )

            # Show pages without page numbers
            missing_pages = [p for p in doc.pages if not p.has_page_number_label]
            if missing_pages:
                print(f"  Pages without page number: {len(missing_pages)}")
                for page in missing_pages[:5]:
                    print(
                        f"    Page {page.page_number}: "
                        f"{page.text_elements} text elements, "
                        f"{page.total_elements} total elements"
                    )
                if len(missing_pages) > 5:
                    print(f"    ... and {len(missing_pages) - 5} more")

        if len(docs_with_issues) > max_docs:
            print(f"\n... and {len(docs_with_issues) - max_docs} more documents")
    else:
        print("\n✓ All documents have 100% page number coverage!")


def main() -> int:
    """Main entry point for the classifier analysis tool.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Analyze classifier performance across all PDFs in data directory."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory containing PDFs (default: data)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-document analysis",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10,
        help="Maximum number of documents to show in detailed report (default: 10)",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        help="Maximum number of PDFs to analyze (for testing)",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 2

    # Find all PDFs
    pdf_files = find_pdfs(args.data_dir)
    if not pdf_files:
        logger.error(f"No PDF files found in {args.data_dir}")
        return 2

    logger.info(f"Found {len(pdf_files)} PDF files")

    # Limit number of PDFs if requested
    if args.max_pdfs:
        pdf_files = pdf_files[: args.max_pdfs]
        logger.info(f"Limiting analysis to {len(pdf_files)} PDFs")

    # Analyze each document
    document_analyses = []
    for pdf_path in pdf_files:
        try:
            doc_analysis = analyze_document(pdf_path)
            document_analyses.append(doc_analysis)
        except Exception as e:
            logger.error(f"Error analyzing {pdf_path}: {e}")
            continue

    # Compute global statistics
    total_pages = sum(doc.total_pages for doc in document_analyses)
    total_pages_with_page_number = sum(
        doc.pages_with_page_number for doc in document_analyses
    )

    global_analysis = GlobalAnalysis(
        documents=document_analyses,
        total_pdfs=len(document_analyses),
        total_pages=total_pages,
        total_pages_with_page_number=total_pages_with_page_number,
    )

    # Print reports
    print_summary_report(global_analysis)

    if args.detailed:
        print_detailed_report(global_analysis, max_docs=args.max_docs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
