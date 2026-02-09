"""Benchmark pipeline for PDF parser evaluation."""
import json
import logging
from datetime import datetime
from datetime import date as date_type
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from ..eval import run_batch_evaluation, EvalPaths
from ..extraction import ParallelSegmentExtractor, SegmentExtractionJob
from ..utilities.base_parser import PDFParser


logger = logging.getLogger(__name__)

# Suppress HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# ========== PYDANTIC MODELS ==========

class BenchmarkResults(BaseModel):
    """Benchmark evaluation results for a parser."""

    date: date_type
    parser_name: str
    benchmark_name: str
    num_pdfs: int
    total_inline_formulas: int
    total_display_formulas: int
    total_tables: int
    # Scores keyed by LLM judge model name
    average_formula_scores: dict[str, float]
    average_inline_formula_scores: dict[str, float]
    average_display_formula_scores: dict[str, float]
    average_table_scores: dict[str, float]
    average_simple_table_scores: dict[str, float]
    average_moderate_table_scores: dict[str, float]
    average_complex_table_scores: dict[str, float]
    # CDM = Character Detection Metrics
    average_cdm_score: float | None = None


# ========== BENCHMARK PIPELINE ==========

class Benchmark:
    """PDF parser benchmark runner."""

    # ========== INITIALIZATION & CONFIGURATION ==========
    def __init__(
        self,
        parser_output_dir: Path,
        ground_truth_dir: Path,
        llm_judge_models: list[str],
        parser: PDFParser,
    ):
        self.parser_output_dir = parser_output_dir
        self.ground_truth_dir = ground_truth_dir
        self.llm_judge_models = llm_judge_models
        self.parser = parser

    def parse(self, pdfs_dir: Path, skip_existing: bool = True) -> None:
        """Parse all PDFs in the specified directory."""
        logger.info("\n🔍 PDF PARSING")

        pdf_files = sorted(pdfs_dir.glob("*.pdf"))

        logger.info(f"   Processing {len(pdf_files)} PDFs")

        for pdf_path in pdf_files:
            output_path = self.parser_output_dir / pdf_path.stem / "parsed.md"

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

    def extract(self, skip_existing: bool = True) -> None:
        """Extract formula and table segments from parsed markdown files."""
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
            return

        logger.info(f"   Processing {len(jobs)} extraction jobs in parallel")

        extractor = ParallelSegmentExtractor(max_workers=20)
        extractor.extract_segments_parallel(jobs)

        logger.info(f"   ✅ Segment extraction completed")

    def evaluate(self, enable_cdm: bool = False, skip_existing: bool = True) -> None:
        """Evaluate parsing results against ground truth."""
        logger.info(f"\n📈 EVALUATION")

        # Collect evaluation jobs for all PDFs with extraction outputs
        jobs = []
        for result_dir in sorted(self.parser_output_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            formulas_path = result_dir / "formulas.json"
            tables_path = result_dir / "tables.json"

            if formulas_path.exists() or tables_path.exists():
                jobs.append(EvalPaths(
                    formulas_path=formulas_path,
                    tables_path=tables_path,
                    cdm_output_dir=result_dir / "cdm",
                ))

        logger.info(f"   Processing {len(jobs)} PDFs")

        run_batch_evaluation(
            llm_judge_models=self.llm_judge_models,
            jobs=jobs,
            enable_cdm=enable_cdm,
            skip_existing=skip_existing,
        )

        logger.info(f"   ✅ Evaluation completed for all PDFs")

    def save_benchmark_summary(self) -> None:
        """Save benchmark summary with aggregated scores to JSON file."""
        logger.info(f"\n💾 SAVING BENCHMARK SUMMARY")

        benchmark_name = self.ground_truth_dir.parent.name
        num_pdfs = 0

        total_inline_formulas = 0
        total_display_formulas = 0
        total_tables = 0

        # scores[category][model] = [score1, score2, ...]
        scores: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        cdm_scores: list[float] = []

        for result_dir in sorted(self.parser_output_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            formulas_path = result_dir / "formulas.json"
            tables_path = result_dir / "tables.json"

            if not formulas_path.exists() and not tables_path.exists():
                continue

            num_pdfs += 1

            if formulas_path.exists():
                with open(formulas_path, 'r', encoding='utf-8') as f:
                    for formula in json.load(f):
                        formula_type = formula["formula_type"]
                        if formula_type == "display-formula":
                            total_display_formulas += 1
                        else:
                            total_inline_formulas += 1

                        for llm_score in formula.get("llm_scores", []):
                            model = llm_score["judge_model"]
                            score = llm_score["score"]
                            scores["formula"][model].append(score)
                            scores[formula_type][model].append(score)

                        cdm = formula.get("cdm_score")
                        if cdm is not None:
                            cdm_scores.append(cdm["score"])

            if tables_path.exists():
                with open(tables_path, 'r', encoding='utf-8') as f:
                    for table in json.load(f):
                        total_tables += 1
                        complexity = table["complexity"]

                        for llm_score in table.get("llm_scores", []):
                            model = llm_score["judge_model"]
                            score = llm_score["score"]
                            scores["table"][model].append(score)
                            scores[complexity][model].append(score)

        def avg(vals: list) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        def avg_by_model(category: str) -> dict[str, float]:
            return {m: avg(vals) for m, vals in scores[category].items()}

        results = BenchmarkResults(
            date=datetime.now().date(),
            parser_name=self.parser.display_name(),
            benchmark_name=benchmark_name,
            num_pdfs=num_pdfs,
            total_inline_formulas=total_inline_formulas,
            total_display_formulas=total_display_formulas,
            total_tables=total_tables,
            average_formula_scores=avg_by_model("formula"),
            average_inline_formula_scores=avg_by_model("inline-formula"),
            average_display_formula_scores=avg_by_model("display-formula"),
            average_table_scores=avg_by_model("table"),
            average_simple_table_scores=avg_by_model("simple"),
            average_moderate_table_scores=avg_by_model("moderate"),
            average_complex_table_scores=avg_by_model("complex"),
            average_cdm_score=avg(cdm_scores) if cdm_scores else None,
        )

        results_path = self.parser_output_dir / "benchmark_results.json"
        results_path.write_text(results.model_dump_json(indent=2))

        logger.info(f"   ✅ Benchmark summary saved to {results_path}")

