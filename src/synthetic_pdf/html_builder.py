"""HTML generation utilities for synthetic PDF content."""
from typing import List, Dict, Any



class HTMLBuilder:
    """Simplified HTML builder for synthetic PDF content."""
    
    def __init__(self, config):
        self.config = config
    
    def build_document(self, data_blocks: List[Dict[str, Any]]) -> str:
        """Build complete HTML document from data blocks."""
        content = ''.join(
            f'<div class="content-block {block["type"]}">{block["data"]}</div>'
            for block in data_blocks
        )
        
        formula_color_css = f"""
        .content-block.formula {{
            color: {self.config.style.formula_color} !important;
        }}""" if self.config.style.formula_color else ""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Parser Benchmark</title>
    <script>
        MathJax = {{
            tex: {{
                displayMath: [['$$', '$$']]
            }},
            svg: {{
                fontCache: 'global'
            }},
            options: {{
                processHtmlClass: 'formula',
                ignoreHtmlClass: 'text'
            }}
        }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    <style>
        @page {{
            size: {self.config.style.pdf_size};
            margin: {self.config.style.pdf_margin_top} {self.config.style.pdf_margin_right} {self.config.style.pdf_margin_bottom} {self.config.style.pdf_margin_left};
        }}

        body {{
            font-family: sans-serif;
            margin: 0;
            padding: 0;
            {self.config.style.to_container_css()}
            {f'background-color: {self.config.style.document_background_color};' if self.config.style.document_background_color else ''}
        }}

        mjx-container[display="true"] {{
            font-size: {self.config.style.formula_font_size} !important;
        }}

        .content-block {{
            {self.config.style.to_css_string()}
        }}{formula_color_css}
    </style>
</head>
<body>
    {content}
</body>
</html>"""
