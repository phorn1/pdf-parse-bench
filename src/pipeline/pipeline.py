"""Benchmark pipeline orchestrator for PDF generation and processing."""
import logging
import random
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
import yaml

from ..synthetic_pdf import StyleConfig, generate_style_combinations, SinglePagePDFGenerator
from ..parser import ParserRegistry
from ..eval import extract_segments_from_md_llm, run_evaluation

logger = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    """Configuration for pipeline execution."""
    generate_pdfs: bool = False
    parse_pdfs: bool = False
    extract_segments: bool = False
    evaluate_results: bool = False
    reuse_timestamp: str | None = None
    llm_judge_model: str

    @model_validator(mode='after')
    def validate_config(self):
        if self.generate_pdfs == bool(self.reuse_timestamp):
            raise ValueError("reuse_timestamp required" if not self.generate_pdfs else "reuse_timestamp cannot be used with generate_pdfs")
        return self


class PdfGenerationConfig(BaseModel):
    """Configuration for synthetic PDF generation."""
    seed: int | None = 10
    random_selection: bool = True
    pdfs_per_config: int = 1
    max_total_pdfs: int = 100


class Config(BaseModel):
    """Main configuration model."""
    pipeline: PipelineConfig
    synthetic_pdf: PdfGenerationConfig
    style: StyleConfig
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
    """Configuration for a single benchmark run with specific style settings."""
    
    name: str = "default"
    timestamp: str
    style: StyleConfig | None = None
    parsers: list[str] = Field(default_factory=list)
    paths: PipelinePaths = Field(default_factory=PipelinePaths, exclude=True)

    @property
    def run_directory(self) -> Path:
        """Directory for this specific run configuration."""
        return self.paths.runs_dir / self.timestamp / self.name

    @property
    def html_output_path(self) -> Path:
        """Path to the generated HTML file."""
        return self.run_directory / "sample.html"

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
        self.paths = PipelinePaths()
        self.config = self._load_config()

    def _load_config(self) -> Config:
        """Load and validate the YAML configuration using Pydantic."""
        with open(self.paths.config_file, encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        return Config(**raw_config)

    def _create_run_configurations(self, timestamp: str) -> list[BenchmarkRunConfig]:
        """Create run configurations based on pipeline settings."""
        style_combinations = generate_style_combinations(self.config.style)
        pdf_config = self.config.synthetic_pdf
        
        run_configs = []
        for i in range(pdf_config.max_total_pdfs):
            if not style_combinations:
                break
                
            if pdf_config.random_selection:
                config_idx = random.randint(0, len(style_combinations) - 1)
                name = f"{i:03d}"
            else:
                config_idx = (i // pdf_config.pdfs_per_config) % len(style_combinations)
                config_instance = (i % pdf_config.pdfs_per_config) + 1
                _, config_name = style_combinations[config_idx]
                name = f"{config_name}_{config_instance}" if pdf_config.pdfs_per_config > 1 else config_name
            
            style, _ = style_combinations[config_idx]
            
            run_config = BenchmarkRunConfig(
                name=name,
                timestamp=timestamp,
                style=style,
                paths=self.paths
            )
            run_configs.append(run_config)
        
        for run_config in run_configs:
            run_config.create_directories()
        
        if run_configs:
            self._update_latest_symlink(run_configs[0].run_directory.parent)
        
        return run_configs
    
    # ========== PDF GENERATION ==========
    
    def _seed_generator(self):
        """Generator for unique configuration seeds."""
        seed = self.config.synthetic_pdf.seed
        random.seed(seed)
        while True:
            yield random.randint(1, 1000000)
    
    async def _generate_pdf_for_config(self, run_config: BenchmarkRunConfig, seed: int) -> None:
        """Generate a single PDF for the given configuration."""
        logger.info(f"Generating PDF for configuration: {run_config.name}")
        
        generator = SinglePagePDFGenerator(
            self.paths.formulas_file, 
            seed,
            run_config.style
        )
        
        await generator.generate_single_page_pdf(
            output_html_path=run_config.html_output_path,
            output_pdf_path=run_config.pdf_output_path,
            output_gt_json=run_config.gt_segments_path,
        )

    # ========== PDF PARSING ==========

    def _parse_pdf_with_parser(self, run_config: BenchmarkRunConfig, parser_name: str) -> None:
        """Parse a single PDF with a specific parser."""
        output_path = run_config.parsed_md_path(parser_name)
        
        # Create parser subdirectory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"  Using parser: {parser_name}")
            ParserRegistry.parse(
                parser_name,
                run_config.pdf_output_path,
                output_path
            )
            logger.info(f"  ✓ Output written to: {output_path}")
        except Exception as e:
            logger.error(f"  ✗ Parser {parser_name} failed for {run_config.name}: {e}")
    
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
                
            logger.info(f"Parsing PDF: {run_config.name}")
            
            for parser_name in enabled_parsers:
                self._parse_pdf_with_parser(run_config, parser_name)
        
        logger.info("Parsing completed for all configurations")
    
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
                        llm_judge_model=self.config.pipeline.llm_judge_model,
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
            seed_gen = self._seed_generator()
            
            logger.info(f"Starting PDF generation for {len(run_configs)} configurations")
            
            for run_config in run_configs:
                await self._generate_pdf_for_config(run_config, next(seed_gen))
            
            logger.info(f"Successfully generated {len(run_configs)} PDFs!")
        else:
            run_configs = self._create_run_configurations(timestamp)
            logger.info("Using existing PDFs - loading configurations")
        
        if pipeline_config.parse_pdfs:
            self.parse_pdfs(run_configs)
        else:
            logger.info("PDF parsing disabled - skipping parsing step")
        
        if pipeline_config.extract_segments:
            logger.info("Extracting segments for evaluation")
            for run_config in run_configs:
                for parser in self.config.parsers:
                    try:
                        extract_segments_from_md_llm(
                            run_config.gt_segments_path,
                            run_config.parsed_md_path(parser), 
                            run_config.segments_json_path(parser)
                        )
                    except Exception as e:
                        logger.error(f"Failed to extract segments for {run_config.name}/{parser}: {e}")
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
