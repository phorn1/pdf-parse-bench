"""Formula validation utilities for synthetic PDF content."""


class FormulaValidator:
    """Validates formula size and layout constraints."""
    
    def __init__(self, pdf_service, html_builder, config):
        self.pdf_service = pdf_service
        self.html_builder = html_builder
        self.config = config
    
    async def validate_formula_size(self, formula: str) -> bool:
        """Validates if a formula fits within page layout constraints using pre-rendering."""
        # Calculate available width based on page size, margins and column layout
        # A4 width = 210mm, subtract margins
        page_width_mm = 210
        margin_left_mm = float(self.config.style.pdf_margin_left.replace('mm', ''))
        margin_right_mm = float(self.config.style.pdf_margin_right.replace('mm', ''))
        content_width_mm = page_width_mm - margin_left_mm - margin_right_mm
        
        # Account for column layout
        actual_columns = self.config.style.column_count
        if actual_columns > 1:
            # For multi-column, divide by columns and subtract column gaps
            column_gap_mm = float(self.config.style.column_gap.replace('mm', ''))
            content_width_mm = (content_width_mm - (actual_columns - 1) * column_gap_mm) / actual_columns
        
        # Convert to pixels and add safety margin
        max_width = content_width_mm * 3.78 * 0.9
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
            margin: {self.config.style.pdf_margin_top} {self.config.style.pdf_margin_right} {self.config.style.pdf_margin_bottom} {self.config.style.pdf_margin_left};
        }}
        body {{ 
            font-family: sans-serif; 
            margin: 10mm;
            width: max-content;
            
        }}
        mjx-container[display="true"] {{ 
            font-size: {self.config.style.formula_font_size} !important;
            width: max-content;
        }}
        .formula {{
            width: max-content;
        }}
    </style>
</head>
<body><div class="formula">$${formula}$$</div></body></html>"""
        
        page = await self.pdf_service._setup_browser_page()
        try:
            await page.set_content(html_content, wait_until='domcontentloaded')
            await self.pdf_service._wait_for_mathjax(page)
            
            # Measure MathJax container dimensions
            bounds = await page.evaluate("""() => {
                const container = document.querySelector('mjx-container');
                if (!container) return null;
                const rect = container.getBoundingClientRect();
                return {width: rect.width, height: rect.height};
            }""")
            
            if not bounds:
                return False
                
            fits = bounds['width'] <= max_width
            max_width_mm = max_width / 3.78
            bounds_width_mm = bounds['width'] / 3.78
            print(f"Formula size: {bounds_width_mm:.1f}mm (max: {max_width_mm:.1f}mm), fits: {fits}")
            return fits
            
        except Exception as e:
            print(f"Formula validation error: {e}")
            return False
        finally:
            await page.close()