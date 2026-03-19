"""Reusable CLI for running benchmark pipelines with different parsers."""

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from .pipeline import Benchmark
from ..utilities.base_parser import PDFParser

console = Console()

# Configure logging with Rich handler
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=False,
        show_path=False,
        markup=True,
    )]
)


# ========== REUSABLE CLI FUNCTION ==========

def run_cli(parser: PDFParser) -> None:
    """
    Run the benchmark CLI with a specific parser.

    This function should be called from a parser's __main__.py module.

    Args:
        parser: The PDF parser instance to use
    """

    @click.command()
    @click.option(
        "-i", "--input-dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=Path("data") / "2025-10-small",
        show_default=True,
        help="Input data directory containing 'pdfs' and 'ground_truth' subdirectories"
    )
    @click.option(
        "-o", "--output-dir",
        type=click.Path(path_type=Path),
        default=None,
        help="Output directory for benchmark results (default: results/{dataset}/{parser})"
    )
    @click.option(
        "--step", "steps",
        multiple=True,
        type=click.Choice(["parse", "extract", "evaluate"], case_sensitive=False),
        help="Steps to run (default: all)"
    )
    @click.option(
        "--reprocess",
        multiple=True,
        type=click.Choice(["all", "parse", "extract", "evaluate"], case_sensitive=False),
        help="Reprocess specific steps (use 'all' for all steps, or specify individual steps)"
    )
    @click.option(
        "--llm-judge-model", "llm_judge_models",
        multiple=True,
        default=("google/gemini-3-flash-preview",),
        show_default=True,
    )
    def benchmark(
        input_dir: Path,
        output_dir: Path,
        steps: tuple[str, ...],
        reprocess: tuple[str, ...],
        llm_judge_models: tuple[str, ...],
    ) -> None:
        f"""Run benchmark pipeline with {parser.display_name()}."""

        # Auto-generate output directory if not specified
        if output_dir is None:
            dataset_name = input_dir.name
            output_dir = Path("results") / dataset_name / parser.parser_id()
            console.print(f"[dim]Output directory: {output_dir}[/dim]")

        # Determine which steps to run (default: all)
        all_steps = {"parse", "extract", "evaluate"}
        active_steps = set(steps) if steps else all_steps

        # Determine which steps to reprocess
        steps_to_reprocess = all_steps if "all" in reprocess else {s.lower() for s in reprocess}

        # Display active steps
        console.print(" → ".join(s for s in ["parse", "extract", "evaluate"] if s in active_steps) or "none")

        # Create benchmark
        pdfs_dir = input_dir / "pdfs"
        ground_truth_dir = input_dir / "ground_truth"

        bench = Benchmark(
            parser_output_dir=output_dir,
            ground_truth_dir=ground_truth_dir,
            llm_judge_models=list(llm_judge_models),
            parser=parser,
        )

        if "parse" in active_steps:
            bench.parse(
                pdfs_dir=pdfs_dir,
                skip_existing="parse" not in steps_to_reprocess
            )

        if "extract" in active_steps:
            bench.extract(skip_existing="extract" not in steps_to_reprocess)

        if "evaluate" in active_steps:
            bench.evaluate(
                skip_existing="evaluate" not in steps_to_reprocess
            )

        bench.save_benchmark_summary()

        console.print(f"\n[bold green]✅ Pipeline completed successfully![/]\n")

    # Run the CLI
    benchmark()
