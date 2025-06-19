"""Single-page PDF generator with iterative content fitting."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
from playwright.async_api import async_playwright, Browser
from pypdf import PdfReader

from config import config
from html_generator import HTMLGenerator


class PDFService:
    """Handles browser management and PDF generation operations."""
    
    def __init__(self):
        self.browser: Browser = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        playwright = await async_playwright().__aenter__()
        self.browser = await playwright.chromium.launch(
            headless=True, 
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.browser:
            await self.browser.close()
    
    async def check_single_page_fit(self, data_blocks: List[Dict[str, Any]], html_generator: HTMLGenerator) -> bool:
        """Checks if given data blocks fit on a single page."""
        html_content = html_generator.generate_html(data_blocks)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_html:
            temp_html.write(html_content)
            temp_html_path = temp_html.name
        
        try:
            page = await self.browser.new_page()
            page.set_default_timeout(10000)
            
            try:
                html_url = Path(temp_html_path).resolve().as_uri()
                await page.goto(html_url, wait_until='networkidle')
                await self._wait_for_mathjax(page)
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                    await page.pdf(
                        path=temp_pdf.name,
                        print_background=True,
                        prefer_css_page_size=True,
                        display_header_footer=False
                    )
                    
                    try:
                        return self._check_pdf_page_count(temp_pdf.name)
                    finally:
                        Path(temp_pdf.name).unlink(missing_ok=True)
            finally:
                await page.close()
                
        finally:
            Path(temp_html_path).unlink(missing_ok=True)
    
    async def generate_pdf_from_html(self, html_path: Path, pdf_path: Path) -> None:
        """Converts HTML to PDF using existing browser instance."""
        page = await self.browser.new_page()
        page.set_default_timeout(10000)
        
        try:
            html_url = html_path.resolve().as_uri()
            await page.goto(html_url, wait_until='networkidle')
            await self._wait_for_mathjax(page)
            
            await page.pdf(
                path=str(pdf_path),
                print_background=True,
                prefer_css_page_size=True,
                display_header_footer=False
            )
        finally:
            await page.close()
    
    @staticmethod
    async def _wait_for_mathjax(page) -> None:
        """Wait for MathJax to complete rendering."""
        await page.wait_for_function(
            "document.querySelector('mjx-container') !== null || "
            "document.querySelector('.MathJax') !== null || "
            "document.documentElement.classList.contains('mjx-processed') || "
            "!document.body.innerHTML.includes('$')",
            timeout=1000
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


class ContentOptimizer:
    """Optimizes content blocks to fit on a single page using binary search."""
    
    def __init__(self, pdf_service: PDFService):
        self.pdf_service = pdf_service
    
    async def find_optimal_blocks(self, all_data: List[Dict[str, Any]], 
                                  start_blocks: int = 4) -> Tuple[List[Dict[str, Any]], int, HTMLGenerator]:
        """Uses binary search to find maximum number of blocks that fit on one page."""
        html_generator = HTMLGenerator()
        html_generator.update_random_page_style()
        
        low = 1
        high = start_blocks * 2
        
        print(f"Testing initial {start_blocks} blocks...")
        start_data = all_data[:start_blocks]
        
        if await self.pdf_service.check_single_page_fit(start_data, html_generator):
            low = start_blocks
            while high <= len(all_data):
                print(f"Testing {high} blocks...")
                if await self.pdf_service.check_single_page_fit(all_data[:high], html_generator):
                    low = high
                    high = min(len(all_data), high * 2)
                else:
                    break
            else:
                return all_data, len(all_data), html_generator
        else:
            high = start_blocks - 1
        
        best_blocks = low
        while low <= high:
            mid = (low + high) // 2
            print(f"Testing {mid} blocks...")
            
            if await self.pdf_service.check_single_page_fit(all_data[:mid], html_generator):
                best_blocks = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return all_data[:best_blocks], best_blocks, html_generator


class SinglePageGenerator:
    """Generates single-page PDFs by iteratively fitting content blocks."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    async def generate_single_page_pdf(self, output_html_path: str,
                                       output_pdf_path: str,
                                       start_blocks: int = 4) -> Tuple[List[Dict[str, Any]], int]:
        """Generates a single-page PDF with optimal content fitting."""
        all_data = self.data
        print(f"Loaded {len(all_data)} data blocks")
        
        async with PDFService() as pdf_service:
            optimizer = ContentOptimizer(pdf_service)
            
            # Find optimal number of blocks
            optimal_data, num_blocks, optimization_html_generator = await optimizer.find_optimal_blocks(all_data, start_blocks)
            print(f"Found optimal fit: {num_blocks} blocks")
            
            # Ensure artifacts directory exists
            config.paths.artifacts_dir.mkdir(exist_ok=True)
            
            # Generate final HTML using the SAME styling as optimization
            html_content = optimization_html_generator.generate_html(optimal_data)
            
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
    with open(config.paths.default_input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    generator = SinglePageGenerator(data=data[19:])
    try:
        used_data, num_blocks = await generator.generate_single_page_pdf(output_html_path=config.paths.default_html_output, output_pdf_path=config.paths.default_pdf_output)
        
        print(f"\nSummary:")
        print(f"- Used {num_blocks} content blocks")
        print(f"- Content blocks used:")
        for i, block in enumerate(used_data):
            content_preview = block['data'][:50] + "..." if len(block['data']) > 50 else block['data']
            print(f"  {i+1}. {content_preview}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())