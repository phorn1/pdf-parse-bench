"""Style generation utilities for synthetic PDF content."""

import random
from typing import Dict
from pydantic import BaseModel

from config import config


class CSSStyle(BaseModel):
    """Represents a CSS style with validation."""
    
    font_family: str
    font_size: str
    font_weight: str
    text_align: str
    color: str
    padding: str
    margin: str
    background_color: str
    break_inside: str = "avoid"
    
    def to_css_string(self) -> str:
        """Convert to CSS rule string."""
        return "; ".join([
            f"font-family: {self.font_family}",
            f"font-size: {self.font_size}",
            f"font-weight: {self.font_weight}",
            f"text-align: {self.text_align}",
            f"color: {self.color}",
            f"padding: {self.padding}",
            f"margin: {self.margin}",
            f"background-color: {self.background_color}",
            f"break-inside: {self.break_inside}",
        ])


class StyleGenerator:
    """Generates random CSS styles for content blocks."""

    def __init__(self):
        self.style_config = config.styles
    
    def generate_random_style(self) -> CSSStyle:
        """Generates a validated CSS style with random properties."""
        return CSSStyle(
            font_family=random.choice(self.style_config.font_families),
            font_size=f"{random.randint(self.style_config.min_font_size, self.style_config.max_font_size)}pt",
            font_weight=random.choice(self.style_config.font_weights),
            text_align=random.choice(self.style_config.text_aligns),
            color=self.style_config.generate_random_readable_color(),
            padding=f"{random.randint(self.style_config.min_padding, self.style_config.max_padding)}px",
            margin=f"{random.randint(self.style_config.min_margin, self.style_config.max_margin)}px",
            background_color=f"hsl(0, 0%, {random.randint(self.style_config.min_background_lightness, self.style_config.max_background_lightness)}%)",
        )
