"""Configuration for the synthetic PDF generator using Pydantic models."""

import random
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the project root directory based on this file's location."""
    return Path(__file__).parent.parent


class PathConfig(BaseModel):
    """File and directory path configuration."""
    
    root: Path = Field(default_factory=get_project_root)
    artifacts_dir: Path = Field(default_factory=lambda: get_project_root() / "artifacts")
    default_input_file: str = Field(default_factory=lambda: str(get_project_root() / "data.json"))
    default_html_output: str = Field(default_factory=lambda: str(get_project_root() / "artifacts" / "benchmark.html"))
    default_pdf_output: str = Field(default_factory=lambda: str(get_project_root() / "artifacts" / "benchmark.pdf"))


class StyleConfig(BaseModel):
    """Styling configuration for content generation."""
    
    font_families: List[str] = [
        "'Times New Roman', Times, serif",
        "'Arial', Helvetica, sans-serif",
        "'Courier New', Courier, monospace",
        "'Georgia', serif",
        "'Verdana', Geneva, sans-serif",
        "'Helvetica Neue', Helvetica, sans-serif",
        "'Calibri', Candara, sans-serif",
        "'Trebuchet MS', sans-serif",
        "'Palatino', 'Palatino Linotype', serif",
        "'Book Antiqua', Palatino, serif",
        "'Garamond', serif",
        "'Century Gothic', sans-serif",
        "'Tahoma', Geneva, sans-serif",
        "'Lucida Sans', sans-serif",
        "'Franklin Gothic Medium', sans-serif",
    ]
    font_weights: List[str] = ["normal", "bold"]
    text_aligns: List[str] = ["justify"]
    def generate_random_readable_color(self) -> str:
        """Generate a random color that ensures good readability.
        
        Uses HSL color space to control lightness for readability.
        Generates colors with low lightness (dark colors) for good contrast
        against light backgrounds.
        """
        # Generate random hue (0-360 degrees)
        hue = random.randint(0, 360)
        # High saturation for vivid colors (40-100%)
        saturation = random.randint(40, 100)
        # Low lightness for dark, readable colors (15-45%)
        lightness = random.randint(15, 45)
        
        return f"hsl({hue}, {saturation}%, {lightness}%)"
    
    min_font_size: int = Field(default=5)
    max_font_size: int = Field(default=10)
    
    min_padding: int = Field(default=5)
    max_padding: int = Field(default=20)
    min_margin: int = Field(default=5)
    max_margin: int = Field(default=25)
    
    min_background_lightness: int = Field(default=95)
    max_background_lightness: int = Field(default=100)

    mathjax_display_font_size: str = "18px"


class LayoutConfig(BaseModel):
    """Layout configuration for document structure."""
    
    min_columns: int = Field(default=2)
    max_columns: int = Field(default=2)
    column_gap: str = "30px"


class PDFConfig(BaseModel):
    """PDF generation configuration."""
    margins: Dict[str, str] = {
        "top": "20mm", 
        "bottom": "20mm", 
        "left": "20mm", 
        "right": "20mm"
    }


class Config(BaseModel):
    """Main configuration container."""
    
    paths: PathConfig = Field(default_factory=PathConfig)
    styles: StyleConfig = Field(default_factory=StyleConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)


# Global configuration instance
config = Config()