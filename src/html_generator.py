"""HTML generation utilities for synthetic PDF content."""

import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from config import config
from style_generator import StyleGenerator, CSSStyle


class PageStyle(BaseModel):
    """Represents page styling configuration."""
    
    column_count: int
    column_gap: str
    content_style: CSSStyle
    
    @property
    def container_css(self) -> str:
        """Get container CSS string."""
        return f"column-count: {self.column_count}; column-gap: {self.column_gap};"


class HTMLTemplate:
    """Handles HTML template generation."""
    
    def __init__(self):
        self.pdf_config = config.pdf
        self.styles_config = config.styles
    
    def build_document(self, content_blocks: List[str], page_style: PageStyle) -> str:
        """Build complete HTML document."""
        return f"""<!DOCTYPE html>
<html lang="de">
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
                ignoreHtmlClass: 'text',
                renderActions: {{
                    addMenu: [0, '', '']
                }}
            }}
        }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    
    <script>
        function scaleOverflowingFormulas() {{
            console.log('Checking formula scaling...');
            const containers = document.querySelectorAll('mjx-container[display="true"]');
            console.log(`Found ${{containers.length}} formula containers`);
            
            containers.forEach((container, index) => {{
                const parent = container.closest('.content-block');
                if (!parent) {{
                    console.log(`Container ${{index}}: No parent found`);
                    return;
                }}
                
                const parentWidth = parent.offsetWidth;
                const containerWidth = container.scrollWidth;
                console.log(`Container ${{index}}: parent=${{parentWidth}}px, formula=${{containerWidth}}px`);
                
                if (containerWidth > parentWidth) {{
                    const scale = (parentWidth / containerWidth) * 0.95; // 5% safety margin
                    container.style.transform = `scale(${{scale}})`;
                    container.style.transformOrigin = 'left top';
                    console.log(`Container ${{index}}: Scaled to ${{scale}}`);
                }} else {{
                    console.log(`Container ${{index}}: No scaling needed`);
                }}
            }});
            
            // Mark as processed for external detection
            document.body.setAttribute('data-formulas-scaled', 'true');
        }}
        
        // Multiple fallback strategies
        function initScaling() {{
            if (window.MathJax && window.MathJax.startup) {{
                MathJax.startup.promise.then(() => {{
                    console.log('MathJax ready via startup promise');
                    setTimeout(scaleOverflowingFormulas, 200);
                }});
            }}
            
            // Fallback: watch for MathJax containers
            const observer = new MutationObserver((mutations) => {{
                const hasFormulas = document.querySelector('mjx-container[display="true"]');
                if (hasFormulas && !document.body.getAttribute('data-formulas-scaled')) {{
                    console.log('MathJax containers detected via observer');
                    setTimeout(scaleOverflowingFormulas, 100);
                    observer.disconnect();
                }}
            }});
            
            observer.observe(document.body, {{ childList: true, subtree: true }});
            
            // Final fallback
            setTimeout(() => {{
                if (!document.body.getAttribute('data-formulas-scaled')) {{
                    console.log('Final fallback scaling');
                    scaleOverflowingFormulas();
                }}
                observer.disconnect();
            }}, 2000);
        }}
        
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initScaling);
        }} else {{
            initScaling();
        }}
    </script>

    <style>
        @page {{
            size: {"A4"};
            margin: {self.pdf_config.margins['top']} {self.pdf_config.margins['right']} {self.pdf_config.margins['bottom']} {self.pdf_config.margins['left']};
        }}

        body {{
            font-family: sans-serif;
            margin: 40px;
            {page_style.container_css}
        }}

        mjx-container[display="true"] {{
            font-size: {self.styles_config.mathjax_display_font_size} !important;
            transform: scale(0.8) !important;
            transform-origin: left top !important;
            overflow: visible;
            display: inline-block;
            max-width: 125% !important;
            box-sizing: border-box;
        }}
        
        mjx-container[display="true"] mjx-math {{
            max-width: 100% !important;
            overflow: visible;
        }}

        .content-block {{
            {page_style.content_style.to_css_string()}
        }}
    </style>
</head>
<body>
    {''.join(content_blocks)}
</body>
</html>"""


class ContentBlockGenerator:
    """Handles content block generation."""
    
    @staticmethod
    def generate_blocks(data_elements: List[Dict[str, Any]]) -> List[str]:
        """Generate HTML blocks from data elements."""
        html_blocks = []
        for item in data_elements:
            css_class = "formula" if item["type"] == "formula" else "text"
            html_blocks.append(f'<div class="content-block {css_class}">{item["data"]}</div>')
        return html_blocks


class HTMLGenerator:
    """Main HTML generator."""

    def __init__(self):
        self.style_generator = StyleGenerator()
        self.html_template = HTMLTemplate()
        self.content_generator = ContentBlockGenerator()
        self.layout_config = config.layout
        self._current_page_style: Optional[PageStyle] = None

    def generate_html(self, data_elements: List[Dict[str, Any]]) -> str:
        """Generate complete HTML content from data elements."""
        if self._current_page_style is None:
            raise ValueError("Page style not initialized. Call update_random_page_style() first.")

        content_blocks = self.content_generator.generate_blocks(data_elements)
        return self.html_template.build_document(content_blocks, self._current_page_style)

    def update_random_page_style(self) -> PageStyle:
        """Generate and set new random page styling."""
        column_count = random.randint(self.layout_config.min_columns, self.layout_config.max_columns)
        content_style = self.style_generator.generate_random_style()
        
        self._current_page_style = PageStyle(
            column_count=column_count,
            column_gap=self.layout_config.column_gap,
            content_style=content_style
        )
        return self._current_page_style
    
    @property
    def current_page_style(self) -> Optional[PageStyle]:
        """Get current page style."""
        return self._current_page_style