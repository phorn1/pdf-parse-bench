"""Benchmark pipeline orchestrator for PDF generation and processing."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, model_validator
import yaml
from tqdm import tqdm

from ..synthetic_pdf import HtmlSinglePagePDFGenerator, HTMLConfig, LaTeXConfig, ParallelLaTeXPDFGenerator, LaTeXPDFJob
from ..parser import ParserRegistry
from ..eval import run_evaluation, ParallelSegmentExtractor, SegmentExtractionJob

logger = logging.getLogger(__name__)

def setup_clean_logging():
    """Suppress noisy third-party loggers."""
    # Suppress noisy external loggers
    noisy_loggers = [
        'httpx', 'urllib3', 'openai._base_client', 'mathpix', 'google_genai',
        'marker', 'mistral', 'llamaparse', 'unstructured', 'transformers',
        'torch', 'tensorflow', 'PIL', 'matplotlib'
    ]
    for name in noisy_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False  # Stop propagation to root logger
    

    # Suppress SSL warnings from third-party libraries
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



class PipelineConfig(BaseModel):
    """Configuration for pipeline execution."""
    generate_pdfs: bool = False
    parse_pdfs: bool = False
    extract_segments: bool = False
    evaluate_results: bool = False
    reuse_timestamp: str | None = None
    formula_llm_judge_model: str

    @model_validator(mode='after')
    def validate_config(self):
        if self.generate_pdfs == bool(self.reuse_timestamp):
            raise ValueError("reuse_timestamp required" if not self.generate_pdfs else "reuse_timestamp cannot be used with generate_pdfs")
        return self


class PdfGenerationConfig(BaseModel):
    """Configuration for synthetic PDF generation."""
    amount: int = 100
    generator_type: Literal["html", "latex"] = "latex"


class Config(BaseModel):
    """Main configuration model."""
    pipeline: PipelineConfig
    synthetic_pdf: PdfGenerationConfig
    parsers: list[str] = Field(default_factory=list)


class PipelinePaths(BaseModel):
    """Centralized path management for the benchmark pipeline."""
    
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    @property
    def formulas_file(self) -> Path:
        return self.project_root / "data" / "formulas.json"
    
    @property
    def config_file(self) -> Path:
        return self.project_root / "config.yaml"
    
    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"
    
    @property
    def runs_dir(self) -> Path:
        return self.artifacts_dir / "runs"
    
    @property
    def latest_symlink(self) -> Path:
        return self.artifacts_dir / "latest"

class BenchmarkRunConfig(BaseModel):
    """Configuration for a single benchmark run."""

    name: str
    timestamp: str
    paths: PipelinePaths
    parsers: list[str] = Field(default_factory=list)

    @property
    def run_directory(self) -> Path:
        """Directory for this specific run configuration."""
        return self.paths.runs_dir / self.timestamp / self.name

    @property
    def html_output_path(self) -> Path:
        """Path to the generated HTML file."""
        return self.run_directory / "sample.html"
    
    @property
    def latex_output_path(self) -> Path:
        """Path to the generated LaTeX file."""
        return self.run_directory / "sample.tex"

    @property
    def pdf_output_path(self) -> Path:
        """Path to the generated PDF file."""
        return self.run_directory / "sample.pdf"

    @property
    def gt_segments_path(self) -> Path:
        """Path to the ground truth JSON file."""
        return self.run_directory / "gt_segments.json"
    
    def parsed_md_path(self, parser_name: str) -> Path:
        """Path to parser output markdown file."""
        return self.run_directory / parser_name / "parsed.md"
    
    def segments_json_path(self, parser_name: str) -> Path:
        """Path to parser matches JSON file."""
        return self.run_directory / parser_name / "segments.json"
    
    def eval_stats_path(self, parser_name: str) -> Path:
        """Path to evaluation statistics JSON file."""
        return self.run_directory / parser_name / "eval_stats.json"
    
    def eval_formula_results_path(self, parser_name: str) -> Path:
        """Path to detailed formula evaluation results JSON file."""
        return self.run_directory / parser_name / "eval_formula_results.json"
    
    def eval_text_results_path(self, parser_name: str) -> Path:
        """Path to detailed text evaluation results JSON file."""
        return self.run_directory / parser_name / "eval_text_results.json"
    
    def create_directories(self) -> None:
        """Create necessary directories for this run."""
        self.run_directory.mkdir(parents=True, exist_ok=True)


class BenchmarkOrchestrator:
    """Main orchestrator for the benchmark pipeline."""
    
    # ========== INITIALIZATION & CONFIGURATION ==========
    
    def __init__(self):
        setup_clean_logging()
        self.paths: PipelinePaths = PipelinePaths()
        self.config: Config= self._load_config()

    def _load_config(self) -> Config:
        """Load and validate the YAML configuration using Pydantic."""
        with open(self.paths.config_file, encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        return Config(**raw_config)

    def _create_run_configurations(self, timestamp: str) -> list[BenchmarkRunConfig]:
        """Create run configurations based on pipeline settings."""
        run_configs = [
            BenchmarkRunConfig(
                name=f"{i:03d}",
                timestamp=timestamp,
                paths=self.paths
            )
            for i in range(self.config.synthetic_pdf.amount)
        ]
        
        for run_config in run_configs:
            run_config.create_directories()
        
        if run_configs:
            self._update_latest_symlink(run_configs[0].run_directory.parent)
        
        return run_configs
    
    # ========== PDF GENERATION ==========
    
    
    async def _generate_html_pdf_for_config(self, run_config: BenchmarkRunConfig) -> None:
        """Generate a single HTML PDF for the given configuration."""
        logger.info(f"Generating HTML PDF for configuration: {run_config.name}")
        
        generator = HtmlSinglePagePDFGenerator(
            self.paths.formulas_file, 
            HTMLConfig.random(),
        )
        
        await generator.generate_single_page_pdf(
            output_html_path=run_config.html_output_path,
            output_pdf_path=run_config.pdf_output_path,
            output_gt_json=run_config.gt_segments_path,
        )

    async def _generate_latex_pdfs(self, run_configs: list[BenchmarkRunConfig]) -> None:
        """Generate LaTeX PDFs in parallel using ParallelLaTeXPDFGenerator."""

        # Prepare PDF tasks with deterministic seeds
        tasks = [
            LaTeXPDFJob(
                config=LaTeXConfig.random(seed=hash(f"{run_config.name}_{run_config.timestamp}")),
                latex_path=run_config.latex_output_path,
                pdf_path=run_config.pdf_output_path,
                gt_path=run_config.gt_segments_path
            )
            for run_config in run_configs
        ]
        
        parallel_generator = ParallelLaTeXPDFGenerator(self.paths.formulas_file)
        
        with tqdm(total=len(tasks), desc="Generating PDFs", unit="pdf") as pbar:
            for _ in parallel_generator.generate_pdfs_parallel(tasks):
                pbar.update(1)
        
        logger.info(f"Successfully generated all {len(tasks)} PDFs")
        

    async def _generate_html_pdfs(self, run_configs: list[BenchmarkRunConfig]) -> None:
        """Generate HTML PDFs sequentially."""
        logger.info(f"Generating {len(run_configs)} HTML PDFs sequentially")
        
        with tqdm(total=len(run_configs), desc="Generating PDFs", unit="pdf") as pbar:
            for run_config in run_configs:
                await self._generate_html_pdf_for_config(run_config)
                pbar.update(1)
        
        logger.info(f"Successfully generated all {len(run_configs)} PDFs")


    # ========== PDF PARSING ==========

    def _parse_pdf_with_parser(self, run_config: BenchmarkRunConfig, parser_name: str) -> None:
        """Parse a single PDF with a specific parser."""
        output_path = run_config.parsed_md_path(parser_name)
        
        # Create parser subdirectory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            ParserRegistry.parse(
                parser_name,
                run_config.pdf_output_path,
                output_path
            )
            logger.info(f"  ✓ {parser_name}")
        except Exception as e:
            logger.error(f"  ✗ {parser_name}: {str(e)[:80]}{'...' if len(str(e)) > 80 else ''}")
    
    def parse_pdfs(self, run_configs: list[BenchmarkRunConfig]) -> None:
        """Parse all PDFs with all configured parsers."""
        enabled_parsers = self.config.parsers
        
        if not enabled_parsers:
            logger.warning("No parsers enabled - skipping parsing step")
            return
            
        logger.info(f"Starting parsing with {len(enabled_parsers)} parsers for {len(run_configs)} PDFs")
        
        for run_config in run_configs:
            if not run_config.pdf_output_path.exists():
                logger.warning(f"PDF not found for {run_config.name} - skipping parsing")
                continue
                
            logger.info(f"Processing PDF {run_config.name}:")
            
            for parser_name in enabled_parsers:
                self._parse_pdf_with_parser(run_config, parser_name)
        
        logger.info("Parsing completed for all configurations")
    
    # ========== SEGMENT EXTRACTION ==========
    
    def _extract_segments_parallel(self, run_configs: list[BenchmarkRunConfig]) -> None:
        """Extract segments in parallel using ParallelSegmentExtractor."""
        enabled_parsers = self.config.parsers
        
        jobs = []
        for run_config in run_configs:
            if not run_config.gt_segments_path.exists():
                logger.warning(f"Ground truth not found for {run_config.name} - skipping")
                continue
                
            for parser in enabled_parsers:
                md_path = run_config.parsed_md_path(parser)
                if md_path.exists():
                    jobs.append(SegmentExtractionJob(
                        gt_json_path=run_config.gt_segments_path,
                        input_md_path=md_path,
                        output_json_path=run_config.segments_json_path(parser)
                    ))
                else:
                    logger.warning(f"Parsed markdown not found for {run_config.name}/{parser}")
        
        if not jobs:
            logger.warning("No segment extraction jobs to process")
            return
        
        logger.info(f"Starting parallel segment extraction for {len(jobs)} jobs")
        
        extractor = ParallelSegmentExtractor(max_workers=10)
        
        with tqdm(total=len(jobs), desc="Extracting segments", unit="job") as pbar:
            for success, message in extractor.extract_segments_parallel(jobs):
                if not success or "⚠" in message:
                    tqdm.write(message)
                pbar.update(1)
        
    
    # ========== EVALUATION ==========
    
    def _evaluate_results(self, run_configs: list[BenchmarkRunConfig]) -> None:
        """Evaluate parsing results against ground truth."""
        enabled_parsers = self.config.parsers
        
        if not enabled_parsers:
            logger.warning("No parsers enabled - skipping evaluation step")
            return
        
        logger.info(f"Starting evaluation for {len(enabled_parsers)} parsers across {len(run_configs)} PDFs")
        
        for run_config in run_configs:
            if not run_config.gt_segments_path.exists():
                logger.warning(f"Ground truth not found for {run_config.name} - skipping evaluation")
                continue
            
            logger.info(f"Evaluating results for: {run_config.name}")
            
            for parser_name in enabled_parsers:
                segments_path = run_config.segments_json_path(parser_name)
                
                if not segments_path.exists():
                    logger.warning(f"  ✗ Segments file not found for {parser_name} - skipping")
                    continue
                
                try:
                    logger.info(f"  Evaluating parser: {parser_name}")
                    
                    run_evaluation(
                        llm_judge_model=self.config.pipeline.formula_llm_judge_model,
                        gt_json_path=run_config.gt_segments_path,
                        parsed_json_path=segments_path,
                        result_stats_path=run_config.eval_stats_path(parser_name),
                        result_formula_evals_path=run_config.eval_formula_results_path(parser_name),
                        result_text_evals_path=run_config.eval_text_results_path(parser_name)
                    )
                    
                    logger.info(f"  ✓ Evaluation completed for {parser_name}")
                    
                except Exception as e:
                    logger.error(f"  ✗ Evaluation failed for {parser_name}: {e}")
        
        logger.info("Evaluation completed for all configurations")
    
    # ========== MAIN PIPELINE ORCHESTRATION ==========
    
    async def run_full_pipeline(self) -> None:
        """Run the complete benchmark pipeline: generate PDFs and parse them."""
        pipeline_config = self.config.pipeline
        
        timestamp = pipeline_config.reuse_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"{'Reusing existing' if pipeline_config.reuse_timestamp else 'Using new'} timestamp: {timestamp}")
        
        if pipeline_config.generate_pdfs:
            run_configs = self._create_run_configurations(timestamp)
            
            logger.info(f"Starting PDF generation for {len(run_configs)} configurations")
            
            match self.config.synthetic_pdf.generator_type:
                case "latex":
                    await self._generate_latex_pdfs(run_configs)
                case "html":
                    await self._generate_html_pdfs(run_configs)

            logger.info(f"Successfully generated {len(run_configs)} PDFs!")
        else:
            run_configs = self._create_run_configurations(timestamp)
            logger.info("Using existing PDFs - loading configurations")
        
        if pipeline_config.parse_pdfs:
            self.parse_pdfs(run_configs)
        else:
            logger.info("PDF parsing disabled - skipping parsing step")
        
        if pipeline_config.extract_segments:
            self._extract_segments_parallel(run_configs)
        else:
            logger.info("Segment extraction disabled - skipping extraction step")
        
        if pipeline_config.evaluate_results:
            self._evaluate_results(run_configs)
        else:
            logger.info("Evaluation disabled - skipping evaluation step")
    
    # ========== UTILITY METHODS ==========
    
    def _update_latest_symlink(self, latest_run_dir: Path) -> None:
        """Create/update symlink to the latest run directory."""
        if self.paths.latest_symlink.is_symlink():
            self.paths.latest_symlink.unlink()
        self.paths.latest_symlink.symlink_to(latest_run_dir, target_is_directory=True)
