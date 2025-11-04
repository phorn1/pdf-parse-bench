"""Benchmark pipeline for PDF parser evaluation."""
import logging
from pathlib import Path

from ..eval import run_evaluation
from ..extraction import ParallelSegmentExtractor, SegmentExtractionJob
from ..utilities.base_parser import PDFParser

logger = logging.getLogger(__name__)

# Suppress HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)


class Benchmark:
    """PDF parser benchmark runner."""

    # ========== INITIALIZATION & CONFIGURATION ==========
    def __init__(
        self,
        parser_output_dir: Path | str,
        ground_truth_dir: Path | str,
    ):
        """
        Initialize benchmark.

        Args:
            parser_output_dir: Directory containing parsed markdown files (or where they will be saved)
            ground_truth_dir: Directory containing ground truth JSON files

        Example (extract and evaluate already parsed results):
            >>> benchmark = Benchmark(
            ...     parser_output_dir="results/my_parser",
            ...     ground_truth_dir="data/2025-10-small/ground_truth"
            ... )
            >>> benchmark.extract().evaluate()

        Example (full pipeline with parsing):
            >>> benchmark = Benchmark(
            ...     parser_output_dir="results/my_parser",
            ...     ground_truth_dir="data/2025-10-small/ground_truth"
            ... )
            >>> benchmark.parse(parser=my_parser, pdfs_dir="data/2025-10-small/pdfs")
            >>> benchmark.extract().evaluate()
        """
        self.parser_output_dir = Path(parser_output_dir)
        self.ground_truth_dir = Path(ground_truth_dir)

    def parse(
        self,
        parser: PDFParser,
        pdfs_dir: Path | str,
        skip_existing: bool = True
    ) -> "Benchmark":
        """
        Parse all PDFs in the specified directory.

        Args:
            parser: The PDF parser to use
            pdfs_dir: Directory containing PDF files to parse
            skip_existing: If True, skip PDFs that already have parsed results

        Returns:
            Self for method chaining
        """
        logger.info("\n🔍 PDF PARSING")

        pdfs_dir = Path(pdfs_dir)
        pdf_files = sorted(pdfs_dir.glob("*.pdf"))

        logger.info(f"   Processing {len(pdf_files)} PDFs")

        for pdf_path in pdf_files:
            output_path = self.parser_output_dir / pdf_path.stem / "parsed.md"

            # Skip if output already exists and skip_existing is True
            if skip_existing and output_path.exists():
                logger.info(f"   ⏩ Parsed MD already exists for {pdf_path.name} - skipping")
                continue

            try:
                parser.parse(pdf_path, output_path)
                logger.info(f"   ✅ {pdf_path.name}")
            except Exception as e:
                logger.warning(f"   ❌ {pdf_path.name}: {e}")

        logger.info("   ✅ Parsing completed")
        return self

    def extract(self, skip_existing: bool = True) -> "Benchmark":
        """
        Extract formula segments from parsed markdown files.

        Args:
            skip_existing: If True, skip extraction for files that already have results

        Returns:
            Self for method chaining
        """
        logger.info(f"\n🧩 SEGMENT EXTRACTION")

        # Collect extraction jobs
        jobs = []
        for result_dir in sorted(self.parser_output_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            # Path to files
            parsed_md_path = result_dir / "parsed.md"
            gt_json_path = self.ground_truth_dir / f"{result_dir.name}.json"
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
            return self

        logger.info(f"   Processing {len(jobs)} extraction jobs in parallel")

        # Run parallel extraction
        extractor = ParallelSegmentExtractor(max_workers=20)
        extractor.extract_segments_parallel(jobs)

        logger.info(f"   ✅ Segment extraction completed")
        return self

    def evaluate(
        self,
        llm_judge_models: str | list[str] = "gpt-5-mini",
        enable_cdm: bool = False,
        skip_existing: bool = True
    ) -> "Benchmark":
        """
        Evaluate parsing results against ground truth.

        Args:
            llm_judge_models: Single model name or list of model names for evaluation
            enable_cdm: If True, enable CDM (Character Detection Metrics) evaluation
            skip_existing: If True, skip evaluation for files that already have results

        Returns:
            Self for method chaining
        """
        logger.info(f"\n📈 EVALUATION")

        # Collect all result directories
        result_dirs = []
        for result_dir in sorted(self.parser_output_dir.iterdir()):
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
        return self


# ========== CONVENIENCE FUNCTION ==========

def run_benchmark(
    parser_output_dir: Path | str,
    ground_truth_dir: Path | str,
) -> Benchmark:
    """
    Quick benchmark runner - runs extract and evaluate on already parsed results.

    Args:
        parser_output_dir: Directory containing parsed markdown files
        ground_truth_dir: Directory containing ground truth JSON files
    """
    return Benchmark(parser_output_dir, ground_truth_dir) \
        .extract() \
        .evaluate()
