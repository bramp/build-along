#!/usr/bin/env python3
"""
Script to analyze sample PDFs and tune configuration parameters.

Run with:
    pants run src/build_a_long/pdf_extract/classifier/tools/tune_config.py
"""

import glob
import json
import logging
import math
import sys
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

# Ensure src is in path for imports if running directly
sys.path.insert(0, "src")

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier import Classifier
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.config import (
    ProgressBarConfig,
    ProgressBarIndicatorConfig,
)
from build_a_long.pdf_extract.extractor.extractor import PageData

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class Tuner(ABC):
    """Abstract base class for configuration tuners."""

    def __init__(self, data_files: list[str]):
        self.data_files = data_files
        self._results_cache: list[tuple[PageData, ClassificationResult]] | None = None

    @abstractmethod
    def tune(self, results: list[tuple[PageData, ClassificationResult]]) -> None:
        """Run the tuning analysis and print recommendations."""
        pass

    def load_results(self) -> list[tuple[PageData, ClassificationResult]]:
        """Load pages and run classification with loose config."""
        if self._results_cache is not None:
            return self._results_cache

        # Create a permissive config to capture all potential candidates
        config = ClassifierConfig()

        # Relax ProgressBarIndicator config to capture outliers
        config.progress_bar_indicator.min_size = 0.1
        config.progress_bar_indicator.max_size = 1000.0
        config.progress_bar_indicator.max_bottom_margin_ratio = 0.5

        classifier = Classifier(config)
        results = []

        print(f"Loading and classifying {len(self.data_files)} files...")
        for file_path in self.data_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)

                pages = data.get("pages", [])
                for page_dict in pages:
                    try:
                        page_data = PageData.model_validate(page_dict)
                        if page_data.bbox:
                            result = classifier.classify(page_data)
                            results.append((page_data, result))
                    except Exception:
                        pass
            except Exception as e:
                log.error(f"Error reading {file_path}: {e}")

        self._results_cache = results
        return results

    def _print_recommendation(
        self,
        config_cls: type[BaseModel],
        field_name: str,
        recommended_value: Any,
        reason: str = "",
    ) -> None:
        """Print the recommendation in the desired format."""
        try:
            field_info = config_cls.model_fields[field_name]
            current_value = field_info.default
        except KeyError:
            current_value = "unknown"

        class_name = config_cls.__name__

        # Color output if possible (basic ANSI)
        green = "\033[92m"
        reset = "\033[0m"

        msg = (
            f"{class_name}.{field_name} = {green}{recommended_value}{reset} "
            f"(was {current_value})"
        )
        if reason:
            msg += f"  # {reason}"
        print(msg)

    def _find_best_indicator(
        self, bar_cand: Candidate, indicators: list[Candidate]
    ) -> Candidate | None:
        """Find the best matching indicator for a progress bar."""
        bar_bbox = bar_cand.bbox
        best_ind_cand = None
        best_dist = float("inf")

        for ind_cand in indicators:
            ind_bbox = ind_cand.bbox

            # 1. Filter by size (indicators are small)
            if ind_bbox.width > 50.0:
                continue
            if ind_bbox.width > bar_bbox.width * 0.2:
                continue

            # 2. Filter by aspect ratio (indicators are roughly square/circular)
            # Avoid divide by zero
            w = max(0.1, ind_bbox.width)
            h = max(0.1, ind_bbox.height)
            aspect = w / h
            if not (0.5 <= aspect <= 2.0):
                continue

            # 3. Vertical alignment check (center to center)
            bar_cy = (bar_bbox.y0 + bar_bbox.y1) / 2
            ind_cy = (ind_bbox.y0 + ind_bbox.y1) / 2
            v_diff = abs(bar_cy - ind_cy)

            # Must be closely aligned (within 1 bar height)
            if v_diff > bar_bbox.height:
                continue

            if v_diff < best_dist:
                best_dist = v_diff
                best_ind_cand = ind_cand

        return best_ind_cand


class ProgressBarTuner(Tuner):
    """Tuner for ProgressBarConfig parameters."""

    def tune(self, results: list[tuple[PageData, ClassificationResult]]) -> None:
        print(f"--- Tuning {ProgressBarConfig.__name__} ---")

        horizontal_excesses: list[float] = []
        vertical_excesses: list[float] = []

        bar_count = 0

        for _, result in results:
            bars = result.get_candidates("progress_bar")
            indicators = result.get_candidates("progress_bar_indicator")

            if not bars:
                continue

            bar_count += len(bars)

            for bar_cand in bars:
                bar_bbox = bar_cand.bbox
                best_ind_cand = self._find_best_indicator(bar_cand, indicators)

                if best_ind_cand:
                    ind_bbox = best_ind_cand.bbox
                    ind_cx = (ind_bbox.x0 + ind_bbox.x1) / 2

                    # Horizontal excess (indicator_search_margin)
                    excess_left = bar_bbox.x0 - ind_cx
                    excess_right = ind_cx - bar_bbox.x1
                    current_h_excess = max(0.0, excess_left, excess_right)

                    # Vertical excess (overlap_expansion_margin)
                    excess_top = max(0.0, bar_bbox.y0 - ind_bbox.y0)
                    excess_bottom = max(0.0, ind_bbox.y1 - bar_bbox.y1)
                    current_v_excess = max(excess_top, excess_bottom)

                    horizontal_excesses.append(current_h_excess)
                    vertical_excesses.append(current_v_excess)

        print(f"Analyzed {bar_count} progress bars.")

        if horizontal_excesses:
            horizontal_excesses.sort()
            p99 = horizontal_excesses[int(len(horizontal_excesses) * 0.99)]
            rec_val = math.ceil(p99 * 1.2)
            self._print_recommendation(
                ProgressBarConfig,
                "indicator_search_margin",
                float(rec_val),
                reason=f"Based on p99 excess of {p99:.2f}px",
            )

        if vertical_excesses:
            vertical_excesses.sort()
            p99 = vertical_excesses[int(len(vertical_excesses) * 0.99)]
            rec_val = math.ceil(p99 * 1.2)
            self._print_recommendation(
                ProgressBarConfig,
                "overlap_expansion_margin",
                float(rec_val),
                reason=f"Based on p99 excess of {p99:.2f}px",
            )
        print("")


class ProgressBarIndicatorTuner(Tuner):
    """Tuner for ProgressBarIndicatorConfig parameters."""

    def tune(self, results: list[tuple[PageData, ClassificationResult]]) -> None:
        print(f"--- Tuning {ProgressBarIndicatorConfig.__name__} ---")

        indicator_min_dims: list[float] = []
        indicator_max_dims: list[float] = []

        matched_indicators_count = 0

        for _, result in results:
            bars = result.get_candidates("progress_bar")
            indicators = result.get_candidates("progress_bar_indicator")

            if not bars:
                continue

            for bar_cand in bars:
                best_ind_cand = self._find_best_indicator(bar_cand, indicators)

                if best_ind_cand:
                    matched_indicators_count += 1
                    bbox = best_ind_cand.bbox
                    indicator_min_dims.append(min(bbox.width, bbox.height))
                    indicator_max_dims.append(max(bbox.width, bbox.height))

        print(f"Found {matched_indicators_count} matched indicators.")

        if indicator_min_dims:
            indicator_min_dims.sort()
            p5 = indicator_min_dims[int(len(indicator_min_dims) * 0.05)]
            rec_val = math.floor(p5 * 0.8)
            self._print_recommendation(
                ProgressBarIndicatorConfig,
                "min_size",
                float(rec_val),
                reason=f"Based on 5th percentile of min(width,height): {p5:.2f}px",
            )

        if indicator_max_dims:
            indicator_max_dims.sort()
            p95 = indicator_max_dims[int(len(indicator_max_dims) * 0.95)]
            rec_val = math.ceil(p95 * 1.2)
            self._print_recommendation(
                ProgressBarIndicatorConfig,
                "max_size",
                float(rec_val),
                reason=f"Based on 95th percentile of max(width,height): {p95:.2f}px",
            )
        print("")


def main():
    files = glob.glob("debug/*_raw.json")
    if not files:
        log.error(
            "No debug/*_raw.json files found. Run extraction with debug output first."
        )
        sys.exit(1)

    # Use first tuner to load results shared by all
    # Actually, Tuner class is abstract.
    # Let's instantiate tuners and manage data loading cleanly.

    tuners = [
        ProgressBarTuner(files),
        ProgressBarIndicatorTuner(files),
    ]

    # Load results once
    # Since Tuner stores cache, and they share the same file list...
    # Actually they are separate instances.
    # Let's just load it once here.

    print("Initializing analysis...")
    # Just use the first tuner to load results for now, effectively sharing logic
    results = tuners[0].load_results()

    for tuner in tuners:
        tuner.tune(results)


if __name__ == "__main__":
    main()
