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

from .latex_config import LaTeXConfig
from .content import create_text_generator, create_formula_generator, load_formulas_from_dataset
from .latex import PageBuilder, compile_latex
from pdf_parse_bench.utilities import FormulaRenderer

logger = logging.getLogger(__name__)


# ========== SINGLE PAGE GENERATION ==========

class SinglePagePDFGenerator:
    """Generates single-page PDFs using LaTeX."""

    def __init__(self, config: LaTeXConfig, formulas: list[str] | None = None):
        """If formulas is None, will download ~35MB dataset."""
        self._config = config
        self._formula_generator = create_formula_generator(seed=config.seed, formulas=formulas)
        self._text_generator = create_text_generator(language=config.language.locale_code, seed=config.seed)

    def generate(self, output_latex_path: Path | None, output_pdf_path: Path, output_gt_json: Path, rendered_formulas_dir: Path | None = None):
        """Generate PDF, ground truth JSON, and optionally save LaTeX source and rendered formulas."""
        page_builder = PageBuilder(
            latex_config=self._config,
            text_generator=self._text_generator,
            formula_generator=self._formula_generator
        )

        page_content = page_builder.generate_page()
        latex_content = page_builder.assemble_latex(page_content)

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
    """Job configuration for parallel PDF generation."""
    config: LaTeXConfig
    latex_path: Path | None
    pdf_path: Path
    gt_path: Path
    rendered_formulas_dir: Path | None = None
    retry_count: int = 0

def _run_pdf_job(job: PDFJob, formulas: list[str]) -> None:
    """Worker function for parallel PDF generation (module-level for pickle)."""
    config = job.config.model_copy(update={"seed": job.config.seed + job.retry_count})
    generator = SinglePagePDFGenerator(config, formulas=formulas)
    generator.generate(job.latex_path, job.pdf_path, job.gt_path, job.rendered_formulas_dir)


class ParallelPDFGenerator:
    """Parallel PDF generator for batch processing."""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)

        self.formulas = load_formulas_from_dataset()

    def generate_pdfs_parallel(self, jobs: list[PDFJob]) -> Iterator[None]:
        """Yields None for each completed job (for progress tracking). Retries failed jobs."""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            pending_jobs = jobs.copy()

            while pending_jobs:
                future_to_job = {
                    executor.submit(_run_pdf_job, job, self.formulas): job
                    for job in pending_jobs
                }
                failed_jobs = []

                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        future.result()
                        if job.retry_count > 0:
                            logger.info(f"Job succeeded on retry {job.retry_count} with seed {job.config.seed}")
                        yield None
                    except Exception as e:
                        job.retry_count += 1
                        self._save_failed_config(job, e)
                        logger.warning(f"Job failed with seed {job.config.seed} (attempt {job.retry_count}): {e}")
                        failed_jobs.append(job)

                if failed_jobs:
                    logger.info(f"Retrying {len(failed_jobs)} failed jobs...")
                pending_jobs = failed_jobs

    @staticmethod
    def _save_failed_config(job: PDFJob, error: Exception) -> None:
        """Save failed configuration for debugging and reproduction."""
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        error_file = debug_dir / f"failed_config_seed_{job.config.seed}.json"

        config_dict = job.config.model_dump(mode='json') | {
            "latex_path": str(job.latex_path) if job.latex_path else None,
            "pdf_path": str(job.pdf_path),
            "gt_path": str(job.gt_path)
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