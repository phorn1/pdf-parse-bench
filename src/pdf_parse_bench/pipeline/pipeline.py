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
    total_tables: int = Field(ge=0, default=0, description="Total number of tables")
    average_formula_scores: dict[str, float] = Field(description="Average formula scores by LLM judge model")
    average_inline_formula_scores: dict[str, float] = Field(description="Average inline formula scores by LLM judge model")
    average_display_formula_scores: dict[str, float] = Field(description="Average display formula scores by LLM judge model")
    average_table_scores: dict[str, float] = Field(default_factory=dict, description="Average table scores by LLM judge model")
    average_simple_table_scores: dict[str, float] = Field(default_factory=dict, description="Average simple table scores by LLM judge model")
    average_moderate_table_scores: dict[str, float] = Field(default_factory=dict, description="Average moderate table scores by LLM judge model")
    average_complex_table_scores: dict[str, float] = Field(default_factory=dict, description="Average complex table scores by LLM judge model")
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
        Extract formula and table segments from parsed markdown files.

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
            output_formulas_json_path = result_dir / "formulas.json"
            output_tables_json_path = result_dir / "tables.json"
            stripped_parsed_text_path = result_dir / "stripped_parsed_text.md"

            if skip_existing:
                with open(gt_json_path, 'r', encoding='utf-8') as f:
                    gt_segments = json.load(f)

                gt_has_formulas = any(s["type"] in ["inline-formula", "display-formula"] for s in gt_segments)
                gt_has_tables = any(s["type"] == "table" for s in gt_segments)

                # Skip if all expected outputs already exist
                formulas_done = not gt_has_formulas or output_formulas_json_path.exists()
                tables_done = not gt_has_tables or output_tables_json_path.exists()

                if formulas_done and tables_done:
                    logger.info(f"   ⏩ Extraction already complete for {result_dir.name} - skipping")
                    continue

            # Create extraction job
            jobs.append(SegmentExtractionJob(
                gt_json_path=gt_json_path,
                input_md_path=parsed_md_path,
                output_formulas_json_path=output_formulas_json_path,
                output_tables_json_path=output_tables_json_path,
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
        llm_judge_models: str | list[str],
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

        # Collect all result directories that have extraction outputs
        result_dirs = []
        for result_dir in sorted(self.parser_output_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            formulas_path = result_dir / "formulas.json"
            tables_path = result_dir / "tables.json"

            if formulas_path.exists() or tables_path.exists():
                result_dirs.append(result_dir)

        logger.info(f"   Processing {len(result_dirs)} PDFs")

        # Evaluate each PDF (run_evaluation handles skip_existing internally)
        for result_dir in result_dirs:
            logger.info(f"   📊 Evaluating {result_dir.name}...")

            try:
                run_evaluation(
                    llm_judge_models=llm_judge_models,
                    formulas_path=result_dir / "formulas.json",
                    tables_path=result_dir / "tables.json",
                    cdm_output_dir=result_dir / "cdm",
                    enable_cdm=enable_cdm,
                    skip_existing=skip_existing,
                )
                logger.info(f"   ✅ {result_dir.name} evaluation completed")
            except Exception as e:
                logger.error(f"   ❌ {result_dir.name} evaluation failed: {e}")

        logger.info(f"   ✅ Evaluation completed for all PDFs")
        return self

    def save_benchmark_summary(self) -> "Benchmark":
        """
        Save benchmark summary to JSON file.

        Aggregates individual scores from formulas.json/tables.json files
        to compute correct global averages (not average of averages).
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

        # Collect individual scores (not per-PDF averages)
        formula_scores: dict[str, list[int]] = defaultdict(list)
        inline_scores: dict[str, list[int]] = defaultdict(list)
        display_scores: dict[str, list[int]] = defaultdict(list)
        table_scores: dict[str, list[int]] = defaultdict(list)
        simple_table_scores: dict[str, list[int]] = defaultdict(list)
        moderate_table_scores: dict[str, list[int]] = defaultdict(list)
        complex_table_scores: dict[str, list[int]] = defaultdict(list)
        cdm_scores: list[float] = []

        for result_dir in sorted(self.parser_output_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            formulas_path = result_dir / "formulas.json"
            tables_path = result_dir / "tables.json"

            if not formulas_path.exists() and not tables_path.exists():
                continue

            num_pdfs += 1

            # Aggregate formula scores
            if formulas_path.exists():
                with open(formulas_path, 'r', encoding='utf-8') as f:
                    for formula in json.load(f):
                        is_display = formula["formula_type"] == "display-formula"

                        for llm_score in formula.get("llm_scores", []):
                            model = llm_score["judge_model"]
                            score = llm_score["score"]
                            formula_scores[model].append(score)
                            if is_display:
                                display_scores[model].append(score)
                            else:
                                inline_scores[model].append(score)

                        cdm = formula.get("cdm_score")
                        if cdm:
                            cdm_scores.append(cdm["score"])

            # Aggregate table scores
            if tables_path.exists():
                with open(tables_path, 'r', encoding='utf-8') as f:
                    for table in json.load(f):
                        complexity = table["complexity"]

                        for llm_score in table.get("llm_scores", []):
                            model = llm_score["judge_model"]
                            score = llm_score["score"]
                            table_scores[model].append(score)

                            if complexity == "simple":
                                simple_table_scores[model].append(score)
                            elif complexity == "moderate":
                                moderate_table_scores[model].append(score)
                            elif complexity == "complex":
                                complex_table_scores[model].append(score)

        # Calculate average scores
        def avg(scores: list) -> float:
            return sum(scores) / len(scores) if scores else 0.0

        def avg_by_model(score_dict: dict[str, list[int]]) -> dict[str, float]:
            return {model: avg(scores) for model, scores in score_dict.items()}

        # Build BenchmarkResults model
        results = BenchmarkResults(
            date=datetime.now().date(),
            parser_name=parser_name,
            benchmark_name=benchmark_name,
            num_pdfs=num_pdfs,
            total_inline_formulas=sum(len(scores) for scores in inline_scores.values()) // max(len(inline_scores), 1),
            total_display_formulas=sum(len(scores) for scores in display_scores.values()) // max(len(display_scores), 1),
            total_tables=sum(len(scores) for scores in table_scores.values()) // max(len(table_scores), 1),
            average_formula_scores=avg_by_model(formula_scores),
            average_inline_formula_scores=avg_by_model(inline_scores),
            average_display_formula_scores=avg_by_model(display_scores),
            average_table_scores=avg_by_model(table_scores),
            average_simple_table_scores=avg_by_model(simple_table_scores),
            average_moderate_table_scores=avg_by_model(moderate_table_scores),
            average_complex_table_scores=avg_by_model(complex_table_scores),
            average_cdm_score=avg(cdm_scores) if cdm_scores else None
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
