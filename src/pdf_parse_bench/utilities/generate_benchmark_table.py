"""
Generate benchmark comparison table from parser evaluation results.

This script reads benchmark_results.json files from all parsers in a results directory
and creates a markdown table comparing their performance.
"""

import argparse
from pathlib import Path

from ..pipeline.pipeline import BenchmarkResults


def collect_benchmark_results(results_dir: Path) -> list[BenchmarkResults]:
    """
    Collect benchmark results from all parsers in the results directory.

    Args:
        results_dir: Path to results directory (e.g., results/2025-q4)

    Returns:
        List of BenchmarkResults objects
    """
    results = []

    for parser_dir in sorted(results_dir.iterdir()):
        if not parser_dir.is_dir():
            continue

        benchmark_results_path = parser_dir / "benchmark_results.json"
        if not benchmark_results_path.exists():
            print(f"Warning: benchmark_results.json not found for {parser_dir.name}")
            continue

        # Load using Pydantic model
        results.append(BenchmarkResults.load_from_file(benchmark_results_path))

    return results


def generate_markdown_table(results: list[BenchmarkResults]) -> str:
    """
    Generate markdown table from benchmark results.

    Args:
        results: List of BenchmarkResults objects

    Returns:
        Markdown table as string
    """
    model = "gpt-5-mini"

    # Sort by average score (descending)
    sorted_results = sorted(
        results,
        key=lambda x: x.average_scores[model],
        reverse=True
    )

    # Build table header
    lines = [
        "| Rank | Parser | Overall | Inline | Display | CDM |",
        "|------|--------|---------|--------|---------|-----|"
    ]

    # Build table rows
    for rank, result in enumerate(sorted_results, start=1):
        avg_score = result.average_scores.get(model, 0.0)
        inline_score = result.average_inline_scores.get(model, 0.0)
        display_score = result.average_display_scores.get(model, 0.0)
        cdm_score = result.average_cdm_score if result.average_cdm_score is not None else "N/A"

        # Format CDM score
        cdm_display = f"{cdm_score:.2f}" if isinstance(cdm_score, float) else cdm_score

        lines.append(
            f"| {rank} | {result.parser_name} | {avg_score:.2f} "
            f"| {inline_score:.2f} | {display_score:.2f} | {cdm_display} |"
        )

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison table from parser evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/2025-q4"),
        help="Path to results directory (default: results/2025-q4)"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    results = collect_benchmark_results(args.results_dir)

    print(f"\nFound {len(results)} parsers with valid results\n")
    print("=" * 80)
    print(generate_markdown_table(results))
    print("=" * 80)
    print("\nCopy the table above into your README.md")


if __name__ == "__main__":
    main()