"""Reusable CLI for running benchmark pipelines with different parsers."""

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from .pipeline import BenchmarkOrchestrator
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
        default=Path("data") / "2025-10-24_12-52-02",
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
        "--only",
        type=click.Choice(["parse", "extract", "evaluate"], case_sensitive=False),
        default=None,
        help="Run only this step"
    )
    @click.option(
        "--skip-parse",
        is_flag=True,
        default=False,
        help="Skip PDF parsing step"
    )
    @click.option(
        "--skip-extract",
        is_flag=True,
        default=False,
        help="Skip formula extraction step"
    )
    @click.option(
        "--skip-evaluate",
        is_flag=True,
        default=False,
        help="Skip evaluation step"
    )
    @click.option(
        "--skip-existing",
        is_flag=True,
        default=False,
        help="Skip already processed files"
    )
    @click.option(
        "--llm-judge-models",
        default="gpt-5-mini",
        show_default=True,
        help="Comma-separated list of LLM models for evaluation (e.g., 'gpt-5-mini,gemini-2.5-flash')"
    )
    @click.option(
        "--enable-cdm/--no-enable-cdm",
        default=False,
        show_default=True,
        help="Enable CDM (Character Detection Metrics) evaluation"
    )
    def benchmark(
        input_dir: Path,
        output_dir: Path,
        only: str | None,
        skip_parse: bool,
        skip_extract: bool,
        skip_evaluate: bool,
        skip_existing: bool,
        llm_judge_models: str,
        enable_cdm: bool,
    ) -> None:
        f"""Run benchmark pipeline with {parser.parser_name()}."""

        # Auto-generate output directory if not specified
        if output_dir is None:
            dataset_name = input_dir.name
            parser_name = parser.parser_name().lower().replace(" ", "_").replace("-", "")
            output_dir = Path("results") / dataset_name / parser_name
            console.print(f"[dim]Output directory: {output_dir}[/dim]")

        # Determine which steps to run
        if only:
            # --only overrides everything: run only the specified step
            run_parse = only == "parse"
            run_extract = only == "extract"
            run_evaluate = only == "evaluate"
        else:
            # Default: run all steps, unless explicitly skipped
            run_parse = not skip_parse
            run_extract = not skip_extract
            run_evaluate = not skip_evaluate

        # Display active steps
        steps = []
        if run_parse:
            steps.append("parse")
        if run_extract:
            steps.append("extract")
        if run_evaluate:
            steps.append("evaluate")
        console.print(" → ".join(steps) if steps else "none")

        # Create orchestrator
        orchestrator = BenchmarkOrchestrator(
            parser=parser,
            input_data_dir=input_dir,
            output_dir=output_dir
        )

        if run_parse:
            orchestrator.parse_pdfs(skip_existing=not skip_existing)

        if run_extract:
            orchestrator.extract_segments(skip_existing=not skip_existing)

        if run_evaluate:
            # Parse LLM judge models
            if "," in llm_judge_models:
                llm_models = [m.strip() for m in llm_judge_models.split(",")]
            else:
                llm_models = llm_judge_models

            orchestrator.evaluate_results(
                llm_judge_models=llm_models,
                enable_cdm=enable_cdm,
                skip_existing=not skip_existing
            )

        console.print(f"\n[bold green]✅ Pipeline completed successfully![/]\n")

    # Run the CLI
    benchmark()