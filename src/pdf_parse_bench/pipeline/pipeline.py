"""Benchmark pipeline for PDF parser evaluation."""
import json
import logging
from datetime import datetime
from datetime import date as date_type
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from ..eval import run_evaluation
from ..extraction import ParallelSegmentExtractor, SegmentExtractionJob
from ..utilities.base_parser import PDFParser


logger = logging.getLogger(__name__)

# Suppress HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# ========== PYDANTIC MODELS ==========

class BenchmarkResults(BaseModel):
    """Benchmark evaluation results for a parser."""

    date: date_type = Field(description="Date of benchmark run")
    parser_name: str = Field(description="Name of the parser")
    benchmark_name: str = Field(description="Name of the benchmark dataset")
    num_pdfs: int = Field(ge=0, description="Number of PDFs evaluated")
    total_inline_formulas: int = Field(ge=0, description="Total number of inline formulas")
    total_display_formulas: int = Field(ge=0, description="Total number of display formulas")
    average_scores: dict[str, float] = Field(description="Average scores by LLM judge model")
    average_inline_scores: dict[str, float] = Field(description="Average inline formula scores by LLM judge model")
    average_display_scores: dict[str, float] = Field(description="Average display formula scores by LLM judge model")
    average_cdm_score: float | None = Field(default=None, description="Average CDM score if available")

    def save_to_file(self, path: Path) -> None:
        """Save model to JSON file with default formatting."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_from_file(cls, path: Path) -> "BenchmarkResults":
        """Load model from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.model_validate_json(f.read())


# ========== BENCHMARK PIPELINE ==========

class Benchmark:
    """PDF parser benchmark runner."""

    # ========== INITIALIZATION & CONFIGURATION ==========
    def __init__(
        self,
        parser_output_dir: Path | str,
        ground_truth_dir: Path | str,
        parser: PDFParser | None = None,
    ):
        """
        Initialize benchmark.

        Args:
            parser_output_dir: Directory containing parsed markdown files (or where they will be saved)
            ground_truth_dir: Directory containing ground truth JSON files
            parser: Optional PDF parser to use for parsing PDFs

        Example (extract and evaluate already parsed results):
            >>> benchmark = Benchmark(
            ...     parser_output_dir="results/my_parser",
            ...     ground_truth_dir="data/2025-10-small/ground_truth"
            ... )
            >>> benchmark.extract().evaluate()

        Example (full pipeline with parsing):
            >>> benchmark = Benchmark(
            ...     parser_output_dir="results/my_parser",
            ...     ground_truth_dir="data/2025-10-small/ground_truth",
            ...     parser=my_parser
            ... )
            >>> benchmark.parse(pdfs_dir="data/2025-10-small/pdfs")
            >>> benchmark.extract().evaluate()
        """
        self.parser_output_dir = Path(parser_output_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self._parser = parser

    def parse(
        self,
        pdfs_dir: Path | str,
        skip_existing: bool = True
    ) -> "Benchmark":
        """
        Parse all PDFs in the specified directory.

        Args:
            pdfs_dir: Directory containing PDF files to parse
            skip_existing: If True, skip PDFs that already have parsed results

        Returns:
            Self for method chaining

        Raises:
            ValueError: If no parser was provided during initialization
        """
        if self._parser is None:
            raise ValueError("No parser provided. Pass a parser to the Benchmark constructor.")

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
                self._parser.parse(pdf_path, output_path)
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

    def save_benchmark_summary(self) -> "Benchmark":
        """
        Save benchmark summary to JSON file.
        """
        logger.info(f"\n💾 SAVING BENCHMARK SUMMARY")

        # Determine parser name
        if self._parser is not None:
            parser_name = self._parser.display_name()
        else:
            parser_name = self.parser_output_dir.name

        # Extract benchmark name from ground_truth_dir
        benchmark_name = self.ground_truth_dir.parent.name

        num_pdfs = 0
        total_inline_formulas = 0
        total_display_formulas = 0
        llm_scores = defaultdict(list)
        llm_inline_scores = defaultdict(list)
        llm_display_scores = defaultdict(list)
        cdm_scores = []

        for result_dir in sorted(self.parser_output_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            eval_stats_path = result_dir / "eval_stats.json"
            formulas_path = result_dir / "formulas.json"

            if not eval_stats_path.exists() or not formulas_path.exists():
                continue

            num_pdfs += 1

            # Count inline and display formulas from formulas.json
            with open(formulas_path, 'r', encoding='utf-8') as f:
                formulas = json.load(f)
                for formula in formulas:
                    gt_formula = formula["gt_formula"]
                    if gt_formula.startswith("$$") and gt_formula.endswith("$$"):
                        total_display_formulas += 1
                    elif gt_formula.startswith("$") and gt_formula.endswith("$"):
                        total_inline_formulas += 1

            # Aggregate scores from eval_stats.json
            with open(eval_stats_path, 'r', encoding='utf-8') as f:
                eval_stats = json.load(f)
                for judge in eval_stats["formula_statistics"]["llm_judge"]:
                    model = judge["judge_model"]
                    llm_scores[model].append(judge["average_score"])
                    llm_inline_scores[model].append(judge["average_inline_score"])
                    llm_display_scores[model].append(judge["average_display_score"])

                # Aggregate CDM scores if available
                cdm_stats = eval_stats["formula_statistics"].get("cdm")
                if cdm_stats:
                    cdm_scores.append(cdm_stats["average_score"])

        # Calculate average scores using helper function
        def avg_scores(score_dict: dict[str, list[float]]) -> dict[str, float]:
            return {
                model: sum(scores) / len(scores) if scores else 0.0
                for model, scores in score_dict.items()
            }

        # Build BenchmarkResults model
        results = BenchmarkResults(
            date=datetime.now().date(),
            parser_name=parser_name,
            benchmark_name=benchmark_name,
            num_pdfs=num_pdfs,
            total_inline_formulas=total_inline_formulas,
            total_display_formulas=total_display_formulas,
            average_scores=avg_scores(llm_scores),
            average_inline_scores=avg_scores(llm_inline_scores),
            average_display_scores=avg_scores(llm_display_scores),
            average_cdm_score=sum(cdm_scores) / len(cdm_scores) if cdm_scores else None
        )

        # Save to JSON file
        results_path = self.parser_output_dir / "benchmark_results.json"
        results.save_to_file(results_path)

        logger.info(f"   ✅ Benchmark summary saved to {results_path}")
        return self


# ========== CONVENIENCE FUNCTION ==========

def run_benchmark(
    parser_output_dir: Path | str,
    ground_truth_dir: Path | str,
) -> Benchmark:
    """
    Quick benchmark runner - runs extract, evaluate and save summary on already parsed results.

    Args:
        parser_output_dir: Directory containing parsed markdown files
        ground_truth_dir: Directory containing ground truth JSON files
    """
    return Benchmark(parser_output_dir, ground_truth_dir) \
        .extract() \
        .evaluate() \
        .save_benchmark_summary()
