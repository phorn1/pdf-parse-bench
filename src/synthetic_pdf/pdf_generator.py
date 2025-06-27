"""Single-page PDF generator with iterative content fitting."""

import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
from playwright.async_api import async_playwright
from pypdf import PdfReader

from config import config
from html_builder import HTMLBuilder


class PDFService:
    """Handles browser management and PDF generation operations."""
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True, 
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.browser.close()
        await self.playwright.stop()

    async def _setup_browser_page(self):
        """Create and configure a new browser page."""
        page = await self.browser.new_page()
        page.set_default_timeout(100000)
        return page

    async def _generate_pdf(self, page, output_path: str):
        """Generate PDF with standard settings."""
        await page.pdf(
            path=output_path,
            print_background=True,
            prefer_css_page_size=True,
            display_header_footer=False
        )

    async def check_single_page_fit(self, data_blocks: List[Dict[str, Any]], html_builder: HTMLBuilder) -> bool:
        """Checks if given data blocks fit on a single page."""
        html_content = html_builder.build_document(data_blocks)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_html:
            temp_html.write(html_content)
            temp_html_path = temp_html.name

        page = await self._setup_browser_page()

        try:
            html_url = Path(temp_html_path).resolve().as_uri()
            await page.goto(html_url, wait_until='domcontentloaded')
            await self._wait_for_mathjax(page)

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                await self._generate_pdf(page, temp_pdf.name)
                result = self._check_pdf_page_count(temp_pdf.name)
                Path(temp_pdf.name).unlink()

            return result
        finally:
            await page.close()
            Path(temp_html_path).unlink()


    async def generate_pdf_from_html(self, html_path: Path, pdf_path: Path) -> None:
        """Converts HTML to PDF using existing browser instance."""
        page = await self._setup_browser_page()
        
        try:
            html_url = html_path.resolve().as_uri()
            await page.goto(html_url, wait_until='domcontentloaded')
            await self._wait_for_mathjax(page)

            await self._generate_pdf(page, str(pdf_path))
        finally:
            await page.close()
    
    @staticmethod
    async def _wait_for_mathjax(page) -> None:
        """Wait for MathJax to complete rendering."""
        # Count expected formulas
        formula_count = await page.evaluate("document.querySelectorAll('.formula').length")
        
        if formula_count > 0:
            # Wait for MathJax containers to appear (any type)
            await page.wait_for_function(
                f"document.querySelectorAll('mjx-container').length >= {formula_count}",
                timeout=10000
            )

    @staticmethod
    def _check_pdf_page_count(pdf_path: str) -> bool:
        """Check if PDF has exactly one page."""
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            page_count = len(pdf_reader.pages)
            is_single_page = page_count == 1
            print(f"PDF has {page_count} page(s), Single page: {is_single_page}")
            return is_single_page




class SinglePageGenerator:
    """Generates single-page PDFs by iteratively fitting content blocks."""
    
    def __init__(self, default_formula_file):
        self.default_formula_file = default_formula_file
        self.data = []
        # Data will be loaded later when needed
    

    async def _try_add_formula_block(self, formula_gen, validator, stats: Dict[str, int]) -> Dict[str, Any]:
        """Try to add a valid formula block."""
        while True:
            formula = next(formula_gen)
            if await validator.validate_formula_size(formula):
                stats['valid_formula_count'] += 1
                return {
                    "type": "formula",
                    "data": f"$${formula}$$"
                }
            else:
                stats['skipped_formula_count'] += 1
                print(f"Skipped oversized formula (total skipped: {stats['skipped_formula_count']})")
    
    def _create_text_block(self, text_gen) -> Dict[str, Any]:
        """Create a text block."""
        return {
            "type": "text",
            "data": next(text_gen)
        }
    
    async def _test_content_fit(self, content_blocks: List[Dict[str, Any]], candidate_block: Dict[str, Any], 
                               pdf_service: PDFService, html_builder: HTMLBuilder) -> bool:
        """Test if adding a candidate block still fits on one page."""
        test_blocks = content_blocks + [candidate_block]
        return await pdf_service.check_single_page_fit(test_blocks, html_builder)

    async def _generate_content_with_fitting(self, pdf_service: PDFService, html_builder: HTMLBuilder) -> List[Dict[str, Any]]:
        """Generate content blocks alternately and check PDF fitting after each addition."""
        from generators import generate_text_paragraphs, load_formula_generator
        from validation import FormulaValidator
        
        # Initialize generators and validator
        text_gen = generate_text_paragraphs()
        formula_gen = load_formula_generator(Path(self.default_formula_file))
        validator = FormulaValidator(pdf_service, html_builder)
        
        content_blocks = []
        stats = {'valid_formula_count': 0, 'skipped_formula_count': 0}
        is_formula_turn = False  # Start with text
        
        while True:
            if is_formula_turn:
                candidate_block = await self._try_add_formula_block(formula_gen, validator, stats)
            else:
                candidate_block = self._create_text_block(text_gen)
            
            if await self._test_content_fit(content_blocks, candidate_block, pdf_service, html_builder):
                content_blocks.append(candidate_block)
                print(f"Added {candidate_block['type']} block (total: {len(content_blocks)})")
                is_formula_turn = not is_formula_turn  # Alternate between text and formula
            else:
                print(f"Reached capacity at {len(content_blocks)} blocks")
                break
        
        print(f"Content generation complete: {len(content_blocks)} blocks total, {stats['valid_formula_count']} formulas, {stats['skipped_formula_count']} formulas skipped")
        return content_blocks


    async def generate_single_page_pdf(self, output_html_path: str,
                                       output_pdf_path: str) -> Tuple[List[Dict[str, Any]], int]:
        """Generates a single-page PDF with optimal content fitting using alternating generation."""
        async with PDFService() as pdf_service:
            # Create HTML generator with configured styling
            html_builder = HTMLBuilder()
            
            # Generate content blocks with immediate fitting validation
            optimal_data = await self._generate_content_with_fitting(pdf_service, html_builder)
            num_blocks = len(optimal_data)
            print(f"Generated optimal content: {num_blocks} blocks")
            
            # Ensure artifacts directory exists
            config.paths.artifacts_dir.mkdir(exist_ok=True)
            
            # Generate final HTML using the configured styling
            html_content = html_builder.build_document(optimal_data)
            
            # Save HTML
            output_html_path = Path(output_html_path)
            with open(output_html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Generated HTML: {output_html_path}")
            
            # Generate PDF
            await pdf_service.generate_pdf_from_html(output_html_path, Path(output_pdf_path))
            print(f"Generated PDF: {output_pdf_path}")
            
            return optimal_data, num_blocks


async def main():
    """Main function for single-page PDF generation."""

    generator = SinglePageGenerator(config.paths.default_formula_file)
    try:
        # Generate paths using the current configuration name
        config_name = config.get_config_name()
        html_path, pdf_path = config.paths.get_output_paths(config_name)
        
        used_data, num_blocks = await generator.generate_single_page_pdf(output_html_path=html_path, output_pdf_path=pdf_path)
        
        print(f"\nSummary:")
        print(f"- Used {num_blocks} content blocks")
        print(f"- Content blocks used:")
        for i, block in enumerate(used_data):
            content_preview = block['data'][:50] + "..." if len(block['data']) > 50 else block['data']
            print(f"  {i+1}. {block['type']}: {content_preview}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())