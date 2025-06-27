"""Configuration for the synthetic PDF generator using Pydantic models."""

import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the project root directory based on this file's location."""
    return Path(__file__).parent.parent


class PathConfig(BaseModel):
    """File and directory path configuration."""
    
    root: Path = Field(default_factory=get_project_root)
    artifacts_dir: Path = Field(default_factory=lambda: get_project_root() / "artifacts")
    default_formula_file: str = Field(default_factory=lambda: str(get_project_root() / "data" / "formulas.json"))
    
    def get_run_directory(self, config_name: str = "default") -> Path:
        """Get the directory for the current run with timestamp and config name."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.artifacts_dir / "runs" / timestamp / config_name
        return run_dir
    
    def get_output_paths(self, config_name: str = "default") -> tuple[str, str]:
        """Get HTML and PDF output paths for the current configuration."""
        run_dir = self.get_run_directory(config_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        html_path = str(run_dir / "benchmark.html")
        pdf_path = str(run_dir / "benchmark.pdf")
        
        # Create/update symlink to latest run
        latest_link = self.artifacts_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.parent, target_is_directory=True)
        
        return html_path, pdf_path


class StyleConfig(BaseModel):
    """Styling configuration for content generation."""
    
    # PDF Layout Settings
    pdf_size: str
    pdf_margin_top: str
    pdf_margin_bottom: str
    pdf_margin_left: str
    pdf_margin_right: str
    
    # Column Layout Settings
    column_count: int
    column_gap: str
    
    # Typography Settings
    font_family: str
    font_size: str
    font_weight: str
    text_align: str
    
    # Color Settings
    text_color: str
    document_background_color: Optional[str] = None
    content_block_background_color: Optional[str] = None
    
    # Spacing Settings
    content_padding: str
    content_margin: str
    
    # Formula Settings
    formula_font_size: str
    formula_color: Optional[str] = None
    
    def to_css_string(self) -> str:
        """Convert style config to CSS string."""
        styles = []
        if self.font_family:
            styles.append(f"font-family: {self.font_family}")
        if self.font_size:
            styles.append(f"font-size: {self.font_size}")
        if self.font_weight:
            styles.append(f"font-weight: {self.font_weight}")
        if self.text_align:
            styles.append(f"text-align: {self.text_align}")
        if self.text_color:
            styles.append(f"color: {self.text_color}")
        if self.content_padding:
            styles.append(f"padding: {self.content_padding}")
        if self.content_margin:
            styles.append(f"margin: {self.content_margin}")
        if self.content_block_background_color:
            styles.append(f"background-color: {self.content_block_background_color}")
        styles.append("break-inside: avoid")
        return "; ".join(styles)
    
    def to_container_css(self) -> str:
        """Get container CSS string for column layout."""
        return f"column-count: {self.column_count}; column-gap: {self.column_gap};"




class Config(BaseModel):
    """Main configuration container."""
    
    paths: PathConfig = Field(default_factory=PathConfig)
    style: StyleConfig
    
    def get_config_name(self) -> str:
        """Generate a descriptive name for the current configuration."""
        font_family = self.style.font_family.split(',')[0].strip().strip("'\"")
        font_size = self.style.font_size.replace('pt', '').replace('px', '')
        return f"config_{font_family.replace(' ', '').lower()}_{font_size}pt"


def load_config() -> Config:
    """Load configuration from YAML file."""
    project_root = get_project_root()
    config_path = project_root / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    return Config(
        style=StyleConfig(**yaml_data['style'])
    )


# Global configuration instance
config = load_config()