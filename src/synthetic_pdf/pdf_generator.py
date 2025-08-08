import json
import logging
from pathlib import Path
from typing import Any

from .style_config import StyleConfig
from .generators import generate_text_paragraphs, load_formula_generator
from .pdf_service import PDFService
from .validation import FormulaSizeValidator

logger = logging.getLogger(__name__)


class SinglePagePDFGenerator:
    """Generates single-page PDFs by iteratively fitting content blocks."""
    
    def __init__(self, default_formula_file: Path, seed: int | None, style: StyleConfig):
        self.style = style
        self.text_gen = generate_text_paragraphs(seed=seed)
        self.formula_gen = load_formula_generator(default_formula_file, seed=seed)

    async def _get_valid_formula_block(self, validator) -> dict[str, str]:
        """Get a valid formula block, skipping oversized ones."""
        while True:
            formula = next(self.formula_gen)
            if await validator.validate_formula_size(formula):
                return {
                    "type": "formula",
                    "data": f"$${formula}$$"
                }
    
    def _create_text_block(self) -> dict[str, str]:
        """Create a text block."""
        return {
            "type": "text",
            "data": next(self.text_gen)
        }
    
    def _build_html_document(self, data_blocks: list[dict[str, str]]) -> str:
        """Build complete HTML document from data blocks."""
        content = ''.join(
            f'<div class="content-block {block["type"]}">{block["data"].replace("<", "&lt;").replace(">", "&gt;") if block["type"] == "formula" else block["data"]}</div>'
            for block in data_blocks
        )

        formula_color_css = f"""
    .content-block.formula {{
        color: {self.style.formula_color} !important;
    }}""" if self.style.formula_color else ""

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
            size: {self.style.pdf_size};
            margin: {self.style.margin_top} {self.style.margin_right} {self.style.margin_bottom} {self.style.margin_left};
        }}

        body {{
            font-family: sans-serif;
            margin: 0;
            padding: 0;
            {self.style.to_container_css()}
            {f'background-color: {self.style.document_background_color};' if self.style.document_background_color else ''}
        }}

        mjx-container[display="true"] {{
            font-size: {self.style.formula_font_size} !important;
        }}

        .content-block {{
            {self.style.to_css_string()}
        }}{formula_color_css}
    </style>
</head>
<body>
    {content}
</body>
</html>"""

    async def _generate_fitting_content(self, pdf_service: PDFService) -> list[dict[str, str]]:
        """Generate content blocks alternately and check PDF fitting after each addition."""

        validator = FormulaSizeValidator(pdf_service, self.style)
        
        content_blocks = []
        is_formula_turn = False  # Start with text
        
        while True:
            if is_formula_turn:
                candidate_block = await self._get_valid_formula_block(validator)
            else:
                candidate_block = self._create_text_block()

            html_content = self._build_html_document(content_blocks + [candidate_block])
            if await pdf_service.check_single_page_fit(html_content):
                content_blocks.append(candidate_block)
                is_formula_turn = not is_formula_turn  # Alternate between text and formula
            else:
                break
        
        logger.debug(f"Content generation complete: {len(content_blocks)} blocks total")
        return content_blocks


    async def generate_single_page_pdf(self, output_html_path: Path,
                                       output_pdf_path: Path, output_gt_json: Path):
        """Generates a single-page PDF with optimal content fitting using alternating generation."""
        async with PDFService() as pdf_service:
            content_blocks = await self._generate_fitting_content(pdf_service)

            # Generate final HTML using the configured styling
            html_content = self._build_html_document(content_blocks)
            
            # Save HTML
            with open(output_html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.debug(f"Generated HTML: {output_html_path}")
            
            # Generate PDF
            await pdf_service.generate_pdf_from_html(html_path=output_html_path, pdf_path=output_pdf_path)
            logger.debug(f"Generated PDF: {output_pdf_path}")

            # Save ground truth JSON
            with open(output_gt_json, 'w', encoding='utf-8') as f:
                json.dump(content_blocks, f, indent=4, ensure_ascii=False)
            logger.debug(f"Generated gt.json: {output_gt_json}")
            
