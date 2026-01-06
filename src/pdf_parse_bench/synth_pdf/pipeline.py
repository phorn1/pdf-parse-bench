"""PDF generation pipeline with parallel processing."""

import json
import logging
import multiprocessing
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .style_config import LaTeXConfig
from .content import create_text_generator, create_formula_generator, load_formulas_from_dataset
from .latex import LaTeXContentGenerator, compile_latex
from pdf_parse_bench.utilities import FormulaRenderer

logger = logging.getLogger(__name__)


# ========== SINGLE PAGE GENERATION ==========

class SinglePagePDFGenerator:
    """Generates single-page PDFs using LaTeX."""

    def __init__(self, config: LaTeXConfig, formulas: list[str] | None = None):
        """If formulas is None, will download ~35MB dataset."""
        self.formula_generator = create_formula_generator(seed=config.seed, formulas=formulas)
        self.text_generator = create_text_generator(language=config.language.locale_code, seed=config.seed)
        self.config = config

    def generate(self, output_latex_path: Path | None, output_pdf_path: Path, output_gt_json: Path, rendered_formulas_dir: Path | None = None):
        """Generate PDF, ground truth JSON, and optionally save LaTeX source and rendered formulas."""
        content_generator = LaTeXContentGenerator(
            config=self.config,
            text_generator=self.text_generator,
            formula_generator=self.formula_generator
        )

        page_content = content_generator.generate_page()
        latex_content = content_generator.assemble_latex(page_content)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_tex_file = Path(temp_dir) / "document.tex"
            temp_tex_file.write_text(latex_content, encoding='utf-8')
            compile_latex(temp_tex_file, output_pdf_path=output_pdf_path)

            if output_latex_path is not None:
                temp_tex_file.rename(output_latex_path)

            gt_data = page_content.to_ground_truth()

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
class PDFJob:
    """Task configuration for parallel PDF generation."""
    config: LaTeXConfig
    latex_path: Path | None
    pdf_path: Path
    gt_path: Path
    rendered_formulas_dir: Path | None = None
    retry_count: int = 0

def _generate_single_pdf_task(task: PDFJob, formulas: list[str]) -> None:
    """Worker function for parallel PDF generation (module-level for pickle)."""
    config = task.config.model_copy(update={"seed": task.config.seed + task.retry_count})
    generator = SinglePagePDFGenerator(config, formulas=formulas)
    generator.generate(task.latex_path, task.pdf_path, task.gt_path, task.rendered_formulas_dir)


class ParallelPDFGenerator:
    """Parallel PDF generator for batch processing."""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)

        self.formulas = load_formulas_from_dataset()

    def generate_pdfs_parallel(self, tasks: list[PDFJob]) -> Iterator[None]:
        """Yields None for each completed task (for progress tracking). Retries failed tasks."""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            pending_tasks = tasks.copy()

            while pending_tasks:
                future_to_task = {
                    executor.submit(_generate_single_pdf_task, task, self.formulas): task
                    for task in pending_tasks
                }
                failed_tasks = []

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

                if failed_tasks:
                    logger.info(f"Retrying {len(failed_tasks)} failed tasks...")
                pending_tasks = failed_tasks

    @staticmethod
    def _save_failed_config(task: PDFJob, error: Exception) -> None:
        """Save failed configuration for debugging and reproduction."""
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        error_file = debug_dir / f"failed_config_seed_{task.config.seed}.json"

        config_dict = task.config.model_dump(mode='json') | {
            "latex_path": str(task.latex_path) if task.latex_path else None,
            "pdf_path": str(task.pdf_path),
            "gt_path": str(task.gt_path)
        }

        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "config": config_dict,
        }

        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Failed configuration saved to {error_file}")