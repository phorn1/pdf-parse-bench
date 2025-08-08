"""Configuration for synthetic PDF generator using Pydantic."""

import re
import random
from itertools import product
from pydantic import BaseModel, field_validator


class StyleConfig(BaseModel):
    """Style configuration for PDF generation."""

    # PDF Layout
    pdf_size: str | list[str]
    margin_top: str | list[str]
    margin_bottom: str | list[str]
    margin_left: str | list[str]
    margin_right: str | list[str]

    # Column Layout
    column_count: int | list[int]
    column_gap: str | list[str]

    # Typography
    font_family: str | list[str]
    font_size: str | list[str]
    font_weight: str | list[str]
    text_align: str | list[str]

    # Colors
    text_color: str | list[str]
    document_background_color: str | list[str] | None = None
    content_block_background_color: str | list[str] | None = None

    # Spacing
    content_padding: str | list[str]
    content_margin: str | list[str]

    # Formula
    formula_font_size: str | list[str]
    formula_color: str | list[str] | None = None

    def to_css_string(self) -> str:
        """Convert to CSS string for content styling."""
        css_map = {
            'font_family': 'font-family',
            'font_size': 'font-size',
            'font_weight': 'font-weight',
            'text_align': 'text-align',
            'text_color': 'color',
            'content_block_background_color': 'background-color',
            'content_padding': 'padding',
            'content_margin': 'margin'
        }

        styles = []
        for field, css_prop in css_map.items():
            value = getattr(self, field)
            if isinstance(value, list):
                raise ValueError(f"List values not supported in CSS generation for field '{field}'")
            styles.append(f"{css_prop}: {value}")

        styles.append("break-inside: avoid")
        return "; ".join(styles)

    def to_container_css(self) -> str:
        """Get container CSS for column layout."""
        if isinstance(self.column_count, list):
            raise ValueError("List values not supported in CSS generation for field 'column_count'")
        if isinstance(self.column_gap, list):
            raise ValueError("List values not supported in CSS generation for field 'column_gap'")
        return f"column-count: {self.column_count}; column-gap: {self.column_gap};"

    @staticmethod
    def _parse_hsl_range(hsl_range: str) -> str:
        """Parse HSL range string and return a random HSL color.
        
        Format: "hsl(h_min-h_max, s_min%-s_max%, l_min%-l_max%)"
        Example: "hsl(0-360, 50-100%, 20-80%)"
        """
        pattern = r'hsl\((\d+)-(\d+),\s*(\d+)-(\d+)%,\s*(\d+)-(\d+)%\)'
        match = re.match(pattern, hsl_range.strip())
        
        h_min, h_max, s_min, s_max, l_min, l_max = map(int, match.groups())
        
        # Generate random values within ranges
        h = random.randint(h_min, h_max)
        s = random.randint(s_min, s_max)
        l = random.randint(l_min, l_max)
        
        return f"hsl({h}, {s}%, {l}%)"
    
    def resolve_color_ranges(self) -> 'StyleConfig':
        """Resolve HSL color ranges to specific colors.
        
        Returns a new StyleConfig instance with resolved colors.
        """
        config_dict = self.model_dump()
        
        # Color fields that support ranges
        color_fields = ['text_color', 'document_background_color', 'content_block_background_color', 'formula_color']
        
        for field in color_fields:
            value = config_dict.get(field)
            if isinstance(value, str) and 'hsl(' in value and '-' in value:
                config_dict[field] = self._parse_hsl_range(value)
            elif isinstance(value, list):
                # Handle lists of colors (some might be ranges)
                resolved_colors = []
                for color in value:
                    if isinstance(color, str) and 'hsl(' in color and '-' in color:
                        resolved_colors.append(self._parse_hsl_range(color))
                    else:
                        resolved_colors.append(color)
                config_dict[field] = resolved_colors
        
        return StyleConfig(**config_dict)


def generate_style_combinations(config: StyleConfig) -> list[tuple[StyleConfig, str]]:
    """Generate all style combinations from a StyleConfig.
    
    Args:
        config: StyleConfig instance with single values or lists for each field
        
    Returns:
        List of (StyleConfig, name) tuples for each combination
    """
    config_dict = config.model_dump()

    # Separate list and single values
    list_fields = {k: v for k, v in config_dict.items() if isinstance(v, list)}
    single_fields = {k: v for k, v in config_dict.items() if not isinstance(v, list)}

    if not list_fields:
        # Even if no list fields, we might have color ranges to resolve
        resolved_config = config.resolve_color_ranges()
        return [(resolved_config, "default")]

    # Generate all combinations
    results = []
    for combination in product(*list_fields.values()):
        # Build config dict for this combination
        config_values = single_fields.copy()
        name_parts = []

        for field_name, value in zip(list_fields.keys(), combination):
            config_values |= {field_name: value}
            clean_value = str(value).replace('mm', '').replace('pt', '').replace('px', '')
            name_parts.append(f"{field_name}_{clean_value}")

        style_config = StyleConfig(**config_values)
        # Resolve color ranges after creating the config
        style_config = style_config.resolve_color_ranges()
        name = "__".join(name_parts)
        results.append((style_config, name))

    return results
