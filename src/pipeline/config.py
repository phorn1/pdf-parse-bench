"""Configuration models and path management for the benchmark pipeline."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, model_validator


# ========== PIPELINE CONFIGURATION ==========

class PipelineConfig(BaseModel):
    """Configuration for pipeline execution."""
    generate_pdfs: bool = False
    parse_pdfs: bool = False
    extract_segments: bool = False
    evaluate_results: bool = False
    reuse_timestamp: str | None = None
    formula_llm_judge_models: list[str]
    enable_cdm_score: bool = False
    enable_formula2png_rendering: bool = False
    enable_png_renderings: bool = False

    @model_validator(mode='after')
    def validate_config(self):
        if self.generate_pdfs == bool(self.reuse_timestamp):
            raise ValueError("reuse_timestamp required" if not self.generate_pdfs else "reuse_timestamp cannot be used with generate_pdfs")
        return self


class PdfGenerationConfig(BaseModel):
    """Configuration for synthetic PDF generation."""
    amount: int
    generator_type: Literal["html", "latex"] = "latex"


class Config(BaseModel):
    """Main configuration model."""
    pipeline: PipelineConfig
    synthetic_pdf: PdfGenerationConfig
    parsers: list[str] = Field(default_factory=list)


# ========== PATH MANAGEMENT ==========

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

    def cdm_image_dir_path(self, parser_name: str) -> Path:
        """Path to CDM image JSON file."""
        return self.run_directory / parser_name / "cdm"

    def rendered_formulas_dir(self, parser_name: str | None = None) -> Path:
        """Path to rendered formula PNG files directory."""
        if parser_name is None:
            return self.run_directory / "rendered_formulas"
        else:
            return self.run_directory / parser_name / "rendered_formulas"

    def create_directories(self) -> None:
        """Create necessary directories for this run."""
        self.run_directory.mkdir(parents=True, exist_ok=True)