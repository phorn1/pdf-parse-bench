"""Single-page PDF generator with iterative content fitting."""

import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
from playwright.async_api import async_playwright
from pypdf import PdfReader

from config import load_all_configs
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
    
    def __init__(self, default_formula_file, config):
        self.default_formula_file = default_formula_file
        self.config = config
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
        text_gen = generate_text_paragraphs(seed=self.config.seed)
        formula_gen = load_formula_generator(Path(self.default_formula_file), seed=self.config.seed)
        validator = FormulaValidator(pdf_service, html_builder, self.config)
        
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
            html_builder = HTMLBuilder(self.config)
            
            # Generate content blocks with immediate fitting validation
            optimal_data = await self._generate_content_with_fitting(pdf_service, html_builder)
            num_blocks = len(optimal_data)
            print(f"Generated optimal content: {num_blocks} blocks")
            
            # Ensure artifacts directory exists
            self.config.paths.artifacts_dir.mkdir(exist_ok=True)
            
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


async def generate_single_config(config_instance, config_name: str) -> Tuple[List[Dict[str, Any]], int]:
    """Generate PDF for a single configuration."""
    generator = SinglePageGenerator(config_instance.paths.formulas_file, config_instance)
    
    # Generate paths using the configuration name
    html_path, pdf_path = config_instance.paths.get_output_paths(config_name)
    
    used_data, num_blocks = await generator.generate_single_page_pdf(
        output_html_path=html_path, 
        output_pdf_path=pdf_path
    )
    
    print(f"\n=== Configuration: {config_name} ===")
    print(f"- Used {num_blocks} content blocks")
    print(f"- Generated: {html_path}")
    print(f"- Generated: {pdf_path}")
    
    return used_data, num_blocks


async def generate_single_config_with_timestamp(config_instance, config_name: str, timestamp: str) -> Tuple[List[Dict[str, Any]], int]:
    """Generate PDF for a single configuration with shared timestamp."""
    generator = SinglePageGenerator(config_instance.paths.formulas_file, config_instance)
    
    # Generate paths using the configuration name and shared timestamp
    run_dir = config_instance.paths.get_run_directory(config_name, timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    html_path = str(run_dir / "benchmark.html")
    pdf_path = str(run_dir / "benchmark.pdf")
    
    # Create/update symlink to latest run
    latest_link = config_instance.paths.artifacts_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.parent, target_is_directory=True)
    
    used_data, num_blocks = await generator.generate_single_page_pdf(
        output_html_path=html_path, 
        output_pdf_path=pdf_path
    )
    
    print(f"\n=== Configuration: {config_name} ===")
    print(f"- Used {num_blocks} content blocks")
    print(f"- Generated: {html_path}")
    print(f"- Generated: {pdf_path}")
    
    return used_data, num_blocks


async def main():
    """Main function for single-page PDF generation with multiple configurations."""
    try:
        # Load all configuration combinations
        all_configs = load_all_configs()
        
        if len(all_configs) == 1:
            # Single configuration
            config_instance, config_name = all_configs[0]
            print(f"Generating single PDF with configuration: {config_name}")
            used_data, num_blocks = await generate_single_config(config_instance, config_name)
            
            print(f"\nSummary:")
            for i, block in enumerate(used_data[:5]):  # Show first 5 blocks
                content_preview = block['data'][:50] + "..." if len(block['data']) > 50 else block['data']
                print(f"  {i+1}. {block['type']}: {content_preview}")
        else:
            # Multiple configurations - use shared timestamp
            from datetime import datetime
            shared_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            print(f"Generating {len(all_configs)} PDFs with different configurations...")
            
            for i, (config_instance, config_name) in enumerate(all_configs, 1):
                print(f"\n[{i}/{len(all_configs)}] Processing: {config_name}")
                await generate_single_config_with_timestamp(config_instance, config_name, shared_timestamp)
            
            print(f"\nSuccessfully generated {len(all_configs)} PDFs!")
            print(f"Output directory: {all_configs[0][0].paths.artifacts_dir / 'latest'}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())