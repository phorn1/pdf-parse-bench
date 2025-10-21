"""Benchmark pipeline orchestrator for PDF generation and processing."""
import logging
from datetime import datetime
from pathlib import Path
import yaml
from tqdm import tqdm

from ..synthetic_pdf import LaTeXConfig, ParallelLaTeXPDFGenerator, LaTeXPDFJob
from ..parser import ParserRegistry
from ..eval import run_evaluation
from ..utilities import ParallelSegmentExtractor, SegmentExtractionJob
from .config import Config, PipelinePaths, BenchmarkRunConfig

logger = logging.getLogger(__name__)

def setup_clean_logging():
    """Suppress noisy third-party loggers."""
    # Suppress noisy external loggers
    noisy_loggers = [
        'httpx', 'urllib3', 'openai._base_client', 'mathpix', 'google_genai',
        'marker', 'mistral', 'llamaparse', 'unstructured', 'transformers',
        'torch', 'tensorflow', 'PIL', 'matplotlib',
    ]
    for name in noisy_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False  # Stop propagation to root logger
    

    # Suppress SSL warnings from third-party libraries
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
    
    
    async def _generate_latex_pdfs(self, run_configs: list[BenchmarkRunConfig]) -> None:
        """Generate LaTeX PDFs in parallel using ParallelLaTeXPDFGenerator."""

        # Prepare PDF tasks with deterministic seeds
        tasks = [
            LaTeXPDFJob(
                config=LaTeXConfig.random(seed=hash(f"{run_config.name}_{run_config.timestamp}")),
                latex_path=run_config.latex_output_path,
                pdf_path=run_config.pdf_output_path,
                gt_path=run_config.gt_segments_path,
                rendered_formulas_dir=run_config.rendered_formulas_dir() if self.config.pipeline.enable_formula2png_rendering else None
            )
            for run_config in run_configs
        ]
        
        parallel_generator = ParallelLaTeXPDFGenerator()
        
        with tqdm(total=len(tasks), desc="   Generating PDFs", unit="pdf") as pbar:
            for _ in parallel_generator.generate_pdfs_parallel(tasks):
                pbar.update(1)
        

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
            logger.info(f"      ✅ {parser_name}")
        except Exception as e:
            logger.error(f"      ❌ {parser_name}: {str(e)}")
    
    def parse_pdfs(self, run_configs: list[BenchmarkRunConfig]) -> None:
        """Parse all PDFs with all configured parsers."""
        enabled_parsers = self.config.parsers
        
        if not enabled_parsers:
            logger.warning("   ⚠️  No parsers enabled - skipping parsing step")
            return
            
        logger.info(f"   Processing {len(run_configs)} PDFs with {len(enabled_parsers)} parsers")
        
        for run_config in run_configs:
            if not run_config.pdf_output_path.exists():
                logger.warning(f"   ⚠️  PDF not found for {run_config.name} - skipping")
                continue
                
            logger.info(f"   📄 Processing PDF {run_config.name}:")
            
            for parser_name in enabled_parsers:
                self._parse_pdf_with_parser(run_config, parser_name)
        
        logger.info(f"   ✅ Parsing completed for all PDFs")
    
    # ========== SEGMENT EXTRACTION ==========
    
    def _extract_segments_parallel(self, run_configs: list[BenchmarkRunConfig], skip_existing=False) -> None:
        """Extract segments in parallel using ParallelSegmentExtractor."""
        enabled_parsers = self.config.parsers
        
        jobs = []
        for run_config in run_configs:
            if not run_config.gt_segments_path.exists():
                logger.warning(f"   ⚠️  Ground truth not found for {run_config.name} - skipping")
                continue
                
            for parser in enabled_parsers:
                md_path = run_config.parsed_md_path(parser)
                extracted_formulas_path = run_config.extracted_formulas_path(parser)
                stripped_parsed_text_path = run_config.stripped_parsed_text_path(parser)
                
                if not md_path.exists():
                    logger.warning(f"   ⚠️  Parsed markdown not found for {run_config.name}/{parser}")
                    continue
                    
                if skip_existing and extracted_formulas_path.exists():
                    logger.info(f"   ⏩ Segments JSON already exists for {run_config.name}/{parser} - skipping")
                    continue
                
                jobs.append(SegmentExtractionJob(
                    gt_json_path=run_config.gt_segments_path,
                    input_md_path=md_path,
                    output_json_path=extracted_formulas_path,
                    stripped_parsed_text_path=stripped_parsed_text_path,
                    rendered_formulas_dir=run_config.rendered_formulas_dir(parser) if self.config.pipeline.enable_formula2png_rendering else None
                ))
        
        if not jobs:
            logger.warning("   ⚠️  No segment extraction jobs to process")
            return
        
        logger.info(f"   Processing {len(jobs)} extraction jobs in parallel")

        extractor = ParallelSegmentExtractor(max_workers=20)
        extractor.extract_segments_parallel(jobs)

        logger.info(f"   ✅ Segment extraction completed")
        
    
    # ========== EVALUATION ==========
    
    def _evaluate_results(self, run_configs: list[BenchmarkRunConfig]) -> None:
        """Evaluate parsing results against ground truth."""
        enabled_parsers = self.config.parsers
        
        if not enabled_parsers:
            logger.warning("   ⚠️  No parsers enabled - skipping evaluation step")
            return
        
        logger.info(f"   Processing {len(run_configs)} PDFs with {len(enabled_parsers)} parsers")
        
        for run_config in run_configs:
            if not run_config.gt_segments_path.exists():
                logger.warning(f"   ⚠️  Ground truth not found for {run_config.name} - skipping")
                continue
            
            logger.info(f"   📊 Evaluating results for PDF {run_config.name}:")
            
            for parser_name in enabled_parsers:
                extracted_formulas_path = run_config.extracted_formulas_path(parser_name)
                
                if not extracted_formulas_path.exists():
                    logger.warning(f"      ⚠️  Segments file not found for {parser_name} - skipping")
                    continue
                
                try:
                    logger.info(f"      🔍 Evaluating {parser_name}...")
                    
                    run_evaluation(
                        llm_judge_models=self.config.pipeline.formula_llm_judge_models,
                        enable_cdm=self.config.pipeline.enable_cdm_score,
                        extracted_formulas_path=extracted_formulas_path,
                        result_stats_path=run_config.eval_stats_path(parser_name),
                        result_formula_evals_path=run_config.eval_formula_results_path(parser_name),
                        cdm_output_dir=run_config.cdm_image_dir_path(parser_name)
                    )
                    
                    logger.info(f"      ✅ {parser_name} evaluation completed")
                    
                except Exception as e:
                    logger.error(f"      ❌ {parser_name} evaluation failed: {e}")
        
        logger.info("   ✅ Evaluation completed for all PDFs")
    
    # ========== MAIN PIPELINE ORCHESTRATION ==========
    
    async def run_full_pipeline(self) -> None:
        """Run the complete benchmark pipeline: generate PDFs and parse them."""
        pipeline_config = self.config.pipeline
        
        timestamp = pipeline_config.reuse_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"🕐 {'Reusing existing' if pipeline_config.reuse_timestamp else 'Starting new'} benchmark run: {timestamp}")
        
        if pipeline_config.generate_pdfs:
            run_configs = self._create_run_configurations(timestamp)
            
            logger.info(f"\n📄 PDF GENERATION")
            logger.info(f"   Generating {len(run_configs)} PDFs using {self.config.synthetic_pdf.generator_type} generator")
            
            match self.config.synthetic_pdf.generator_type:
                case "latex":
                    await self._generate_latex_pdfs(run_configs)

            logger.info(f"   ✅ Successfully generated {len(run_configs)} PDFs")
        else:
            run_configs = self._create_run_configurations(timestamp)
            logger.info(f"\n📄 PDF GENERATION")
            logger.info(f"   ⏩ Skipped - using existing PDFs from {timestamp}")
        
        if pipeline_config.parse_pdfs:
            logger.info(f"\n🔍 PDF PARSING")
            self.parse_pdfs(run_configs)
        else:
            logger.info(f"\n🔍 PDF PARSING")
            logger.info(f"   ⏩ Skipped - parsing disabled")
        
        if pipeline_config.extract_segments:
            logger.info(f"\n🧩 SEGMENT EXTRACTION")
            self._extract_segments_parallel(run_configs)
        else:
            logger.info(f"\n🧩 SEGMENT EXTRACTION")
            logger.info(f"   ⏩ Skipped - extraction disabled")
        
        if pipeline_config.evaluate_results:
            logger.info(f"\n📈 EVALUATION")
            self._evaluate_results(run_configs)
        else:
            logger.info(f"\n📈 EVALUATION")
            logger.info(f"   ⏩ Skipped - evaluation disabled")
        
        logger.info(f"\n🎉 Pipeline completed successfully!")
    
    # ========== UTILITY METHODS ==========
    
    def _update_latest_symlink(self, latest_run_dir: Path) -> None:
        """Create/update symlink to the latest run directory."""
        if self.paths.latest_symlink.is_symlink():
            self.paths.latest_symlink.unlink()
        self.paths.latest_symlink.symlink_to(latest_run_dir, target_is_directory=True)
