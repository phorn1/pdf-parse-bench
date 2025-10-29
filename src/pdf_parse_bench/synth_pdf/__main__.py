"""CLI entry point for synthetic PDF generation.

This module provides a command-line interface for generating synthetic benchmark PDFs
using the LaTeX-based PDF generation system.
"""

from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

from .pdf_generator import ParallelLaTeXPDFGenerator, LaTeXPDFJob
from .style_config import LaTeXConfig

console = Console()


# ========== MAIN CLI COMMAND ==========

@click.command()
@click.option(
    "-n", "--num-pdfs",
    type=int,
    required=True,
    help="Number of PDFs to generate",
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data"),
    show_default=True,
    help="Base output directory for generated PDFs and metadata",
)
@click.option(
    "--render-formulas",
    is_flag=True,
    help="Render formulas to PNG images (requires LaTeX)",
)
@click.option(
    "--save-latex",
    is_flag=True,
    help="Save LaTeX source files (for debugging purposes)",
)
@click.option(
    "--no-timestamp",
    is_flag=True,
    help="Disable automatic timestamped subdirectory creation",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Maximum number of parallel workers (default: CPU count - 1)",
)
def generate(
    output_dir: Path,
    num_pdfs: int,
    render_formulas: bool,
    save_latex: bool,
    no_timestamp: bool,
    max_workers: int | None,
) -> None:
    """Generate synthetic benchmark PDFs with LaTeX."""
    # Generate timestamp for seed generation and directory naming
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Handle timestamp-based output directory (enabled by default)
    if not no_timestamp:
        output_dir = output_dir / timestamp

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(exist_ok=True)

    gt_dir = output_dir / "ground_truth"
    gt_dir.mkdir(exist_ok=True)

    if save_latex:
        latex_dir = output_dir / "latex"
        latex_dir.mkdir(exist_ok=True)
    else:
        latex_dir = None

    # Optional formula rendering
    if render_formulas:
        formulas_dir = output_dir / "rendered_formulas"
        formulas_dir.mkdir(exist_ok=True)
    else:
        formulas_dir = None

    # Prepare PDF generation tasks with deterministic seeds
    tasks = []
    for i in range(num_pdfs):
        doc_name = f"{i:03d}"

        # Create deterministic seed from doc name and timestamp
        task_seed = hash(f"{doc_name}_{timestamp}")

        # Create individual formula rendering directory if enabled
        if formulas_dir:
            doc_formulas_dir = formulas_dir / doc_name
            doc_formulas_dir.mkdir(exist_ok=True)
        else:
            doc_formulas_dir = None

        task = LaTeXPDFJob(
            config=LaTeXConfig.random(seed=task_seed),
            latex_path=latex_dir / f"{doc_name}.tex" if latex_dir else None,
            pdf_path=pdf_dir / f"{doc_name}.pdf",
            gt_path=gt_dir / f"{doc_name}.json",
            rendered_formulas_dir=doc_formulas_dir
        )
        tasks.append(task)

    # Generate PDFs in parallel with rich progress bar
    console.print(f"\n[bold green]🚀 Starting parallel PDF generation into directory {output_dir}[/]")
    generator = ParallelLaTeXPDFGenerator(max_workers=max_workers)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Generating PDFs...", total=len(tasks))

        for _ in generator.generate_pdfs_parallel(tasks):
            progress.update(task, advance=1)

    console.print(f"\n[bold green]✅ Successfully generated {num_pdfs} PDFs[/]")


if __name__ == "__main__":
    generate()