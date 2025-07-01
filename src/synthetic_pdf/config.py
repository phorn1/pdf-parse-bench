"""Configuration for the synthetic PDF generator using Pydantic models."""

import yaml
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Dict, Optional, Union, List, Any, Tuple
from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the project root directory based on this file's location."""
    return Path(__file__).parent.parent.parent


class PathConfig(BaseModel):
    """File and directory path configuration."""
    
    root: Path = Field(default_factory=get_project_root)
    artifacts_dir: Path = Field(default_factory=lambda: get_project_root() / "artifacts")
    formulas_file: str = Field(default_factory=lambda: str(get_project_root() / "data" / "formulas.json"))
    
    def get_run_directory(self, config_name: str = "default", timestamp: str = None) -> Path:
        """Get the directory for the current run with timestamp and config name."""
        if timestamp is None:
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
    pdf_size: Union[str, List[str]]
    pdf_margin_top: Union[str, List[str]]
    pdf_margin_bottom: Union[str, List[str]]
    pdf_margin_left: Union[str, List[str]]
    pdf_margin_right: Union[str, List[str]]
    
    # Column Layout Settings
    column_count: Union[int, List[int]]
    column_gap: Union[str, List[str]]
    
    # Typography Settings
    font_family: Union[str, List[str]]
    font_size: Union[str, List[str]]
    font_weight: Union[str, List[str]]
    text_align: Union[str, List[str]]
    
    # Color Settings
    text_color: Union[str, List[str]]
    document_background_color: Union[str, List[str], None] = None
    content_block_background_color: Union[str, List[str], None] = None
    
    # Spacing Settings
    content_padding: Union[str, List[str]]
    content_margin: Union[str, List[str]]
    
    # Formula Settings
    formula_font_size: Union[str, List[str]]
    formula_color: Union[str, List[str], None] = None
    
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
    
    seed: Optional[int] = None
    paths: PathConfig = Field(default_factory=PathConfig)
    style: StyleConfig
    

def _normalize_config_value(value: Union[Any, List[Any]]) -> List[Any]:
    """Normalize a config value to always be a list."""
    if isinstance(value, list):
        return value
    return [value]


def _generate_config_combinations(style_data: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str]]:
    """Generate all possible combinations from style configuration with list values."""
    # Extract all fields that have list values
    list_fields = {}
    single_fields = {}
    
    for key, value in style_data.items():
        if isinstance(value, list):
            list_fields[key] = value
        else:
            single_fields[key] = value
    
    if not list_fields:
        # No list fields, return single configuration
        return [(style_data, "default")]
    
    # Generate all combinations of list fields
    field_names = list(list_fields.keys())
    field_values = list(list_fields.values())
    
    combinations = []
    for combination in product(*field_values):
        # Create configuration for this combination
        config_dict = single_fields.copy()
        config_name_parts = []
        
        for field_name, field_value in zip(field_names, combination):
            config_dict[field_name] = field_value
            # Create readable name part
            if isinstance(field_value, str):
                clean_value = field_value.replace('mm', '').replace('pt', '').replace('px', '')
                config_name_parts.append(f"{field_name}_{clean_value}")
            else:
                config_name_parts.append(f"{field_name}_{field_value}")
        
        config_name = "__".join(config_name_parts)
        combinations.append((config_dict, config_name))
    
    return combinations



def load_all_configs() -> List[Tuple[Config, str]]:
    """Load all configuration combinations from YAML file."""
    project_root = get_project_root()
    config_path = project_root / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    combinations = _generate_config_combinations(yaml_data['style'])
    
    configs = []
    for style_dict, config_name in combinations:
        config = Config(
            seed=yaml_data.get('seed'),
            style=StyleConfig(**style_dict)
        )
        configs.append((config, config_name))
    
    return configs

