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
from .content_generator import LaTeXContentGenerator
from .compiler import LaTeXCompiler
from ..generators import generate_text_paragraphs, load_formula_generator

logger = logging.getLogger(__name__)



class LaTeXSinglePagePDFGenerator:
    """Generates single-page PDFs using LaTeX."""
    
    def __init__(self, default_formula_file: Path, config: LaTeXConfig):
        """Initialize the LaTeX PDF generator to match HTML interface.
        
        Args:
            default_formula_file: Path to formulas JSON file
            config: Configuration for LaTeX document generation
        """
        self.formula_generator = load_formula_generator(default_formula_file, seed=config.seed)
        self.text_generator = generate_text_paragraphs(language=config.language.locale_code, seed=config.seed)
        self.config = config

    def generate_single_page_pdf(self, output_latex_path: Path, output_pdf_path: Path, output_gt_json: Path):
        """Generate a single-page PDF with LaTeX to match HTML interface.
        
        Args:
            output_latex_path: Path for the generated LaTeX file
            output_pdf_path: Path for the generated PDF file
            output_gt_json: Path for the ground truth JSON file
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

            # Copy LaTeX file to output path
            temp_tex_file.rename(output_latex_path)

            # Save ground truth JSON
            gt_data = page_content.to_ground_truth()
            with open(output_gt_json, 'w', encoding='utf-8') as f:
                json.dump(gt_data, f, indent=4, ensure_ascii=False)


# ========== PARALLEL PDF GENERATION ==========

@dataclass
class LaTeXPDFJob:
    """Task configuration for parallel PDF generation."""
    config: LaTeXConfig
    latex_path: Path
    pdf_path: Path
    gt_path: Path

def _generate_single_pdf_task(formula_file: Path, task: LaTeXPDFJob) -> None:
    """Worker function for parallel PDF generation."""
    generator = LaTeXSinglePagePDFGenerator(formula_file, task.config)
    generator.generate_single_page_pdf(task.latex_path, task.pdf_path, task.gt_path)


class ParallelLaTeXPDFGenerator:
    """Parallel PDF generator for batch processing."""
    
    def __init__(self, formula_file: Path, max_workers: int | None = None):
        """Initialize parallel PDF generator.
        
        Args:
            formula_file: Path to formulas JSON file
            max_workers: Number of parallel workers (defaults to CPU count - 1)
            debug_dir: Directory to save failed configurations for debugging
        """
        self.formula_file = formula_file
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)

    def generate_pdfs_parallel(self, tasks: list[LaTeXPDFJob]) -> Iterator[None]:
        """Generate multiple PDFs in parallel.
        
        Args:
            tasks: List of PDFTask configurations
            
        Yields:
            None for each completed task (used for progress tracking)
        """
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(_generate_single_pdf_task, self.formula_file, task): task 
                for task in tasks
            }
            
            # Yield completion signals as tasks complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()  # This will raise any exception from the worker
                    yield None
                except Exception as e:
                    self._save_failed_config(task, e)
                    logger.error(f"Task failed with seed {task.config.seed}: {e}")
                    yield None

    @staticmethod
    def _save_failed_config(task: LaTeXPDFJob, error: Exception) -> None:
        """Save failed configuration for debugging and reproduction."""
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        error_file = debug_dir / f"failed_config_seed_{task.config.seed}.json"

        config_dict = task.config.model_dump(mode='json')
        # Convert Path objects to strings for JSON serialization
        config_dict.update({
            "latex_path": str(task.latex_path),
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

