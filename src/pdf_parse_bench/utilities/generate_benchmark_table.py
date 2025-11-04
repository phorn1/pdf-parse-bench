"""
Generate benchmark comparison table from parser evaluation results.

This script reads eval_stats.json files from all parsers in a results directory
and creates a markdown table comparing their performance.
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass


# ========== PARSER NAME MAPPING ==========
PARSER_NAMES = {
    "deepseek_ocr": "DeepSeek-OCR",
    "dots_ocr": "dots.ocr",
    "gemini_2_5_flash": "Gemini 2.5 Flash",
    "gemini_2_5_pro": "Gemini 2.5 Pro",
    "got_ocr2": "GOT-OCR2.0",
    "gpt_5_mini": "GPT-5 mini",
    "gpt_5_nano": "GPT-5 nano",
    "llamaparse": "LlamaParse",
    "mathpix": "Mathpix",
    "mineru": "MinerU2.5",
    "mistral": "Mistral OCR",
    "monkey_ocr": "MonkeyOCR-pro-3B",
    "nanonetsocrs": "Nanonets-OCR-s",
    "olmocr": "olmOCR-2-7B-1025-FP8",
    "paddle_ocr_vl": "PaddleOCR-VL",
    "pp_structurev3": "PP-StructureV3",
    "pymupdf4llm": "PyMuPDF4LLM",
}


@dataclass
class ParserStats:
    """Statistics for a single parser."""

    name: str
    average_score: float
    accuracy_percentage: float
    inline_score: float
    display_score: float
    num_samples: int

    def __post_init__(self):
        """Round values for cleaner display."""
        self.average_score = round(self.average_score, 2)
        self.accuracy_percentage = round(self.accuracy_percentage, 2)
        self.inline_score = round(self.inline_score, 2)
        self.display_score = round(self.display_score, 2)


def collect_parser_stats(results_dir: Path, max_samples: int | None = None) -> list[ParserStats]:
    """
    Collect statistics from all parsers in the results directory.

    Args:
        results_dir: Path to results directory (e.g., results/2025-10-v1)
        max_samples: Maximum number of PDF samples to include (uses first N in sorted order)

    Returns:
        List of ParserStats objects
    """
    parser_stats = []

    for parser_dir in sorted(results_dir.iterdir()):
        if not parser_dir.is_dir():
            continue

        # Collect all eval_stats.json files for this parser
        eval_files = list(parser_dir.glob("*/eval_stats.json"))

        if not eval_files:
            print(f"Warning: No eval_stats.json files found in {parser_dir.name}")
            continue

        # Sort by PDF number (000, 001, 002, ...) and limit if requested
        eval_files = sorted(eval_files, key=lambda x: x.parent.name)
        if max_samples is not None:
            eval_files = eval_files[:max_samples]

        # Aggregate scores
        total_avg_score = 0.0
        total_accuracy = 0.0
        total_inline = 0.0
        total_display = 0.0
        valid_samples = 0

        for eval_file in eval_files:
            try:
                with open(eval_file) as f:
                    data = json.load(f)

                # Extract llm_judge data
                llm_judge = data.get("formula_statistics", {}).get("llm_judge", [])
                if not llm_judge:
                    continue

                judge_data = llm_judge[0]
                total_avg_score += judge_data.get("average_score", 0)
                total_accuracy += judge_data.get("accuracy_percentage", 0)
                total_inline += judge_data.get("average_inline_score", 0)
                total_display += judge_data.get("average_display_score", 0)
                valid_samples += 1

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Warning: Could not parse {eval_file}: {e}")
                continue

        if valid_samples == 0:
            print(f"Warning: No valid samples found for {parser_dir.name}")
            continue

        # Calculate averages
        parser_stats.append(ParserStats(
            name=PARSER_NAMES.get(parser_dir.name, parser_dir.name),
            average_score=total_avg_score / valid_samples,
            accuracy_percentage=total_accuracy / valid_samples,
            inline_score=total_inline / valid_samples,
            display_score=total_display / valid_samples,
            num_samples=valid_samples
        ))

    return parser_stats


def generate_markdown_table(stats: list[ParserStats]) -> str:
    """
    Generate markdown table from parser statistics.

    Args:
        stats: List of ParserStats objects

    Returns:
        Markdown table as string
    """
    # Sort by average_score (descending)
    sorted_stats = sorted(stats, key=lambda x: x.average_score, reverse=True)

    # Build table
    lines = [
        "| Rank | Parser | Avg Score | Accuracy (%) | Inline Score | Display Score | Samples |",
        "|------|--------|-----------|--------------|--------------|---------------|---------|"
    ]

    for rank, stat in enumerate(sorted_stats, start=1):
        lines.append(
            f"| {rank} | {stat.name} | {stat.average_score} | {stat.accuracy_percentage}% "
            f"| {stat.inline_score} | {stat.display_score} | {stat.num_samples} |"
        )

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison table from parser evaluation results"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Maximum number of PDF samples to include (uses first N in sorted order, e.g., 000-024 for --samples 25)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/2025-10-v1"),
        help="Path to results directory (default: results/2025-10-v1)"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    sample_info = f"all samples" if args.samples is None else f"first {args.samples} samples"
    print(f"Collecting statistics from {args.results_dir} ({sample_info})...")
    stats = collect_parser_stats(args.results_dir, max_samples=args.samples)

    if not stats:
        print("Error: No parser statistics collected")
        return

    print(f"\nFound {len(stats)} parsers with valid results\n")
    print("=" * 80)
    print(generate_markdown_table(stats))
    print("=" * 80)
    print("\nCopy the table above into your README.md")


if __name__ == "__main__":
    main()