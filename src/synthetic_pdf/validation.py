"""Formula validation utilities for synthetic PDF content."""
import logging
from .pdf_service import PDFService
from .style_config import StyleConfig

logger = logging.getLogger(__name__)



class FormulaSizeValidator:
    """Validates formula size and layout constraints."""
    
    def __init__(self, pdf_service: PDFService, style: StyleConfig):
        self.pdf_service = pdf_service
        self.style = style
    
    async def validate_formula_size(self, formula: str) -> bool:
        """Validates if a formula fits within page layout constraints using pre-rendering."""
        # Calculate available width based on page size, margins and column layout
        # A4 width = 210mm, subtract margins
        page_width_mm = 210
        margin_left_mm = float(self.style.margin_left.replace('mm', ''))
        margin_right_mm = float(self.style.margin_right.replace('mm', ''))
        content_width_mm = page_width_mm - margin_left_mm - margin_right_mm
        
        # Account for column layout
        actual_columns = self.style.column_count
        if actual_columns > 1:
            # For multi-column, divide by columns and subtract column gaps
            column_gap_mm = float(self.style.column_gap.replace('mm', ''))
            content_width_mm = (content_width_mm - (actual_columns - 1) * column_gap_mm) / actual_columns
        
        # Convert to pixels and add safety margin
        max_width = content_width_mm * 3.78
        
        # Escape HTML characters in formula to prevent DOM parsing issues
        escaped_formula = formula.replace('<', '&lt;').replace('>', '&gt;')
        
        html_content = f"""<!DOCTYPE html>
<html><head>
    <meta charset="UTF-8">
    <script>
        MathJax = {{
            tex: {{
                displayMath: [['$$', '$$']]
            }},
            svg: {{
                fontCache: 'global'
            }}
        }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    <style>
        @page {{
            size: A4;
            margin: {self.style.margin_top} {self.style.margin_right} {self.style.margin_bottom} {self.style.margin_left};
        }}
        body {{ 
            font-family: sans-serif; 
            width: max-content;
        }}
        mjx-container[display="true"] {{ 
            font-size: {self.style.formula_font_size} !important;
            width: max-content;
        }}
        .formula {{
            width: max-content;
        }}
    </style>
</head>
<body><div class="formula">$${escaped_formula}$$</div></body></html>"""
        
        page = await self.pdf_service.setup_browser_page()
        try:
            await page.set_content(html_content, wait_until='domcontentloaded')
            await self.pdf_service.wait_for_mathjax(page)

            # Measure MathJax container dimensions
            if not (bounds := await page.evaluate("""() => {
                const container = document.querySelector('mjx-container');
                if (!container) return null;
                const rect = container.getBoundingClientRect();
                return {width: rect.width, height: rect.height};
            }""")):
                return False

            fits = bounds['width'] <= max_width
            max_width_mm = max_width / 3.78
            bounds_width_mm = bounds['width'] / 3.78
            logger.debug(f"Formula size: {bounds_width_mm:.1f}mm (max: {max_width_mm:.1f}mm), fits: {fits}")
            return fits
        finally:
            await page.close()