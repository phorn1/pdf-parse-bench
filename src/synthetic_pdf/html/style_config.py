"""Configuration for synthetic PDF generator using Pydantic."""

import random
from pydantic import BaseModel


class HTMLConfig(BaseModel):
    """HTML Style configuration for PDF generation."""

    # PDF Layout
    pdf_size: str = "A4"
    margin_top: str = "20mm"
    margin_bottom: str = "20mm"
    margin_left: str = "20mm"
    margin_right: str = "20mm"

    # Column Layout
    column_count: int = 2
    column_gap: str = "10mm"

    # Typography
    font_family: str = "'Times New Roman', Times, serif"
    font_size: str = "12pt"
    font_weight: str = "normal"
    text_align: str = "justify"

    # Colors
    text_color: str = "black"
    document_background_color: str | None = "white"
    content_block_background_color: str | None = None

    # Spacing
    content_padding: str = "2mm"
    content_margin: str = "2mm"

    # Formula
    formula_font_size: str = "10pt"
    formula_color: str | None = "black"

    @classmethod
    def random(cls) -> 'HTMLConfig':
        """Generate a random StyleConfig with randomized values."""
        font_families = [
            "'Times New Roman', Times, serif",
            "Arial, sans-serif", 
            "'Helvetica Neue', Helvetica, Arial, sans-serif",
            "Georgia, 'Times New Roman', Times, serif",
            "'Courier New', Courier, monospace",
            "Verdana, Geneva, sans-serif",
            "Tahoma, Geneva, sans-serif",
            "'Trebuchet MS', Helvetica, sans-serif",
            "Impact, Charcoal, sans-serif",
            "'Palatino Linotype', 'Book Antiqua', Palatino, serif"
        ]
        
        margins = ["10mm", "20mm", "30mm"]
        column_gaps = ["10mm", "20mm"]
        font_sizes = ["10pt", "12pt"]
        
        def random_hsl_color(h_min: int, h_max: int, s_min: int, s_max: int, l_min: int, l_max: int) -> str:
            h = random.randint(h_min, h_max)
            s = random.randint(s_min, s_max)
            l = random.randint(l_min, l_max)
            return f"hsl({h}, {s}%, {l}%)"
        
        return cls(
            pdf_size="A4",
            margin_top=random.choice(margins),
            margin_bottom=random.choice(margins),
            margin_left=random.choice(margins),
            margin_right=random.choice(margins),
            column_count=random.choice([2]),
            column_gap=random.choice(column_gaps),
            font_family=random.choice(font_families),
            font_size=random.choice(font_sizes),
            font_weight="normal",
            text_align="justify",
            text_color=random_hsl_color(0, 360, 100, 100, 0, 0),
            document_background_color=random_hsl_color(0, 360, 10, 30, 95, 95) if random.random() > 0.1 else None,
            content_block_background_color=random_hsl_color(0, 360, 5, 20, 55, 75) if random.random() > 0.1 else None,
            content_padding="2mm",
            content_margin="2mm",
            formula_font_size="10pt",
            formula_color="black"
        )

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
            if value is not None:
                styles.append(f"{css_prop}: {value}")

        styles.append("break-inside: avoid")
        return "; ".join(styles)

    def to_container_css(self) -> str:
        """Get container CSS for column layout."""
        return f"column-count: {self.column_count}; column-gap: {self.column_gap};"

