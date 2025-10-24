"""Main LaTeX PDF generator with automated variations."""

import json
import tempfile
import traceback
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator
from dataclasses import dataclass
import multiprocessing

from .style_config import LaTeXConfig
from .assembler import LaTeXContentGenerator
from .compiler import LaTeXCompiler
from .random_content import generate_text_paragraphs, load_formula_generator, load_formulas_from_dataset
from pdf_benchmark.utilities import FormulaRenderer

logger = logging.getLogger(__name__)



class LaTeXSinglePagePDFGenerator:
    """Generates single-page PDFs using LaTeX."""

    def __init__(self, config: LaTeXConfig, formulas: list[str] | None = None):
        """Initialize the LaTeX PDF generator to match HTML interface.

        Args:
            config: Configuration for LaTeX document generation
            formulas: Pre-loaded formulas list (if None, will download ~35MB dataset)
        """
        self.formula_generator = load_formula_generator(seed=config.seed, formulas=formulas)
        self.text_generator = generate_text_paragraphs(language=config.language.locale_code, seed=config.seed)
        self.config = config

    def generate_single_page_pdf(self, output_latex_path: Path | None, output_pdf_path: Path, output_gt_json: Path, rendered_formulas_dir: Path | None = None):
        """Generate a single-page PDF with LaTeX to match HTML interface.

        Args:
            output_latex_path: Optional path for the generated LaTeX file (None to skip saving)
            output_pdf_path: Path for the generated PDF file
            output_gt_json: Path for the ground truth JSON file
            rendered_formulas_dir: Optional directory to save rendered formula PNGs
        """
        # Build LaTeX content generator with generators
        builder = LaTeXContentGenerator(
            config=self.config,
            text_generator=self.text_generator,
            formula_generator=self.formula_generator
        )
        
        # Generate page content
        page_content = builder.generate_page_content()
        
        # Build LaTeX document
        latex_content = builder.template.build_document_template(page_content)
        
        # Use temporary directory for compilation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_tex_file = temp_path / "document.tex"
            
            # Write .tex file to temp directory
            temp_tex_file.write_text(latex_content, encoding='utf-8')
            
            # Compile to PDF directly to output path
            LaTeXCompiler.compile_latex(temp_tex_file, output_pdf_path=output_pdf_path)

            # Copy LaTeX file to output path (if requested)
            if output_latex_path is not None:
                temp_tex_file.rename(output_latex_path)

            # Save ground truth JSON
            gt_data = page_content.to_ground_truth()

            # Render formulas if requested
            if rendered_formulas_dir is not None:
                renderer = FormulaRenderer()
                for i, segment in enumerate(seg for seg in gt_data if seg["type"] in ["inline-formula", "display-formula"]):
                    segment["rendered_png"] = renderer.render_formula(
                        segment["data"],
                        rendered_formulas_dir,
                        f"formula_{i:03d}"
                    )

            with open(output_gt_json, 'w', encoding='utf-8') as f:
                json.dump(gt_data, f, indent=4, ensure_ascii=False)


# ========== PARALLEL PDF GENERATION ==========

@dataclass
class LaTeXPDFJob:
    """Task configuration for parallel PDF generation."""
    config: LaTeXConfig
    latex_path: Path | None
    pdf_path: Path
    gt_path: Path
    rendered_formulas_dir: Path | None = None
    retry_count: int = 0

def _generate_single_pdf_task(task: LaTeXPDFJob, formulas: list[str]) -> None:
    """Worker function for parallel PDF generation.

    Args:
        task: Task configuration
        formulas: Pre-loaded formulas to avoid re-downloading in each worker
    """
    # Update seed to include retry count for different random content on retry
    task.config.seed = task.config.seed + task.retry_count

    generator = LaTeXSinglePagePDFGenerator(task.config, formulas=formulas)
    generator.generate_single_page_pdf(task.latex_path, task.pdf_path, task.gt_path, task.rendered_formulas_dir)


class ParallelLaTeXPDFGenerator:
    """Parallel PDF generator for batch processing."""

    def __init__(self, max_workers: int | None = None):
        """Initialize parallel PDF generator.

        Args:
            max_workers: Number of parallel workers (defaults to CPU count - 1)
        """
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)

        self.formulas = load_formulas_from_dataset()

    def generate_pdfs_parallel(self, tasks: list[LaTeXPDFJob]) -> Iterator[None]:
        """Generate multiple PDFs in parallel with infinite retry.
        
        Args:
            tasks: List of PDFTask configurations
            
        Yields:
            None for each completed task (used for progress tracking)
        """
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            pending_tasks = tasks.copy()
            
            while pending_tasks:
                # Submit current batch of tasks
                future_to_task = {
                    executor.submit(_generate_single_pdf_task, task, self.formulas): task
                    for task in pending_tasks
                }
                
                failed_tasks = []
                
                # Process completed tasks
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        future.result()
                        if task.retry_count > 0:
                            logger.info(f"Task succeeded on retry {task.retry_count} with seed {task.config.seed}")
                        yield None
                    except Exception as e:
                        task.retry_count += 1
                        self._save_failed_config(task, e)
                        logger.warning(f"Task failed with seed {task.config.seed} (attempt {task.retry_count}): {e}")
                        failed_tasks.append(task)
                
                # Continue with failed tasks
                if failed_tasks:
                    logger.info(f"Retrying {len(failed_tasks)} failed tasks...")
                    pending_tasks = failed_tasks
                else:
                    pending_tasks = []

    @staticmethod
    def _save_failed_config(task: LaTeXPDFJob, error: Exception) -> None:
        """Save failed configuration for debugging and reproduction."""
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        error_file = debug_dir / f"failed_config_seed_{task.config.seed}.json"

        config_dict = task.config.model_dump(mode='json')
        # Convert Path objects to strings for JSON serialization
        config_dict.update({
            "latex_path": str(task.latex_path) if task.latex_path else None,
            "pdf_path": str(task.pdf_path),
            "gt_path": str(task.gt_path)
        })

        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "config": config_dict,
        }

        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Failed configuration saved to {error_file}")

