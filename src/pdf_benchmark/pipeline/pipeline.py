"""Benchmark pipeline orchestrator for PDF generation and processing."""
import logging
from pathlib import Path

from ..eval import run_evaluation
from ..extraction import ParallelSegmentExtractor, SegmentExtractionJob
from ..utilities.base_parser import PDFParser

logger = logging.getLogger(__name__)

# Suppress HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# def setup_clean_logging():
#     """Suppress noisy third-party loggers."""
#     # Suppress noisy external loggers
#     noisy_loggers = [
#         'httpx', 'urllib3', 'openai._base_client', 'mathpix', 'google_genai',
#         'marker', 'mistral', 'llamaparse', 'unstructured', 'transformers',
#         'torch', 'tensorflow', 'PIL', 'matplotlib',
#     ]
#     for name in noisy_loggers:
#         logger = logging.getLogger(name)
#         logger.setLevel(logging.WARNING)
#         logger.propagate = False  # Stop propagation to root logger
#
#     # Suppress SSL warnings from third-party libraries
#     import urllib3
#     urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BenchmarkOrchestrator:
    """Main orchestrator for the benchmark pipeline."""
    
    # ========== INITIALIZATION & CONFIGURATION ==========
    def __init__(self, parser: PDFParser, input_data_dir: Path, output_dir: Path):
        self.parser = parser
        self.input_data_dir = input_data_dir
        self.output_dir = output_dir

    def parse_pdfs(self, skip_existing: bool = True):
        """Parse all PDFs in input directory."""
        logger.info("\n🔍 PDF PARSING")

        pdf_dir = self.input_data_dir / "pdfs"
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

        logger.info(f"   Processing {len(pdf_files)} PDFs")

        for pdf_path in pdf_files:
            output_path = self.output_dir / pdf_path.stem / "parsed.md"

            # Skip if output already exists and skip_existing is True
            if skip_existing and output_path.exists():
                logger.info(f"   ⏩ Parsed MD already exists for {pdf_path.name} - skipping")
                continue

            try:
                self.parser.parse(pdf_path, output_path)
                logger.info(f"   ✅ {pdf_path.name}")
            except Exception as e:
                logger.warning(f"   ❌ {pdf_path.name}: {e}")

        logger.info("   ✅ Parsing completed")

    def extract_segments(self, skip_existing=True) -> None:
        """Extract formula segments from parsed markdown files."""
        logger.info(f"\n🧩 SEGMENT EXTRACTION")

        # Collect extraction jobs
        jobs = []
        for result_dir in sorted(self.output_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            # Path to files
            parsed_md_path = result_dir / "parsed.md"
            gt_json_path = self.input_data_dir / "ground_truth" / f"{result_dir.name}.json"
            output_json_path = result_dir / "formulas.json"
            stripped_parsed_text_path = result_dir / "stripped_parsed_text.md"

            # Skip if output already exists and skip_existing is True
            if skip_existing and output_json_path.exists():
                logger.info(f"   ⏩ Formulas JSON already exists for {result_dir.name} - skipping")
                continue

            # Create extraction job
            jobs.append(SegmentExtractionJob(
                gt_json_path=gt_json_path,
                input_md_path=parsed_md_path,
                output_json_path=output_json_path,
                stripped_parsed_text_path=stripped_parsed_text_path,
                rendered_formulas_dir=None
            ))

        if not jobs:
            logger.warning("   ⚠️  No segment extraction jobs to process")
            return

        logger.info(f"   Processing {len(jobs)} extraction jobs in parallel")

        # Run parallel extraction
        extractor = ParallelSegmentExtractor(max_workers=20)
        extractor.extract_segments_parallel(jobs)

        logger.info(f"   ✅ Segment extraction completed")

    def evaluate_results(self, llm_judge_models: str | list[str] = "gpt-5-mini", enable_cdm: bool = False, skip_existing: bool = True) -> None:
        """Evaluate parsing results against ground truth."""
        logger.info(f"\n📈 EVALUATION")

        # Collect all result directories
        result_dirs = []
        for result_dir in sorted(self.output_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            # Check if formulas.json exists (required for evaluation)
            formulas_path = result_dir / "formulas.json"
            if not formulas_path.exists():
                logger.warning(f"   ⚠️  Formulas file not found for {result_dir.name} - skipping")
                continue

            # Check if evaluation already exists
            eval_stats_path = result_dir / "eval_stats.json"
            if skip_existing and eval_stats_path.exists():
                logger.info(f"   ⏩ Evaluation already exists for {result_dir.name} - skipping")
                continue

            result_dirs.append(result_dir)

        logger.info(f"   Processing {len(result_dirs)} PDFs")

        # Evaluate each PDF
        for result_dir in result_dirs:
            logger.info(f"   📊 Evaluating {result_dir.name}...")

            # Define paths
            extracted_formulas_path = result_dir / "formulas.json"
            eval_stats_path = result_dir / "eval_stats.json"
            eval_formula_results_path = result_dir / "eval_formula_results.json"
            cdm_output_dir = result_dir / "cdm"

            try:
                run_evaluation(
                    llm_judge_models=llm_judge_models,
                    enable_cdm=enable_cdm,
                    skip_existing=skip_existing,
                    extracted_formulas_path=extracted_formulas_path,
                    result_stats_path=eval_stats_path,
                    result_formula_evals_path=eval_formula_results_path,
                    cdm_output_dir=cdm_output_dir
                )
                logger.info(f"   ✅ {result_dir.name} evaluation completed")
            except Exception as e:
                logger.error(f"   ❌ {result_dir.name} evaluation failed: {e}")

        logger.info(f"   ✅ Evaluation completed for all PDFs")
