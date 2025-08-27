import tempfile
import logging
from pathlib import Path
from playwright.async_api import async_playwright, Page
from pypdf import PdfReader

logger = logging.getLogger(__name__)


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

    async def setup_browser_page(self) -> Page:
        """Create and configure a new browser page."""
        page = await self.browser.new_page()
        page.set_default_timeout(100000)
        return page

    async def _generate_pdf(self, page: Page, output_path: Path) -> None:
        """Generate PDF with standard settings."""
        await page.pdf(
            path=str(output_path),
            print_background=True,
            prefer_css_page_size=True,
            display_header_footer=False
        )

    async def check_single_page_fit(self, html_content: str) -> bool:
        """Checks if given data blocks fit on a single page."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_html:
            temp_html.write(html_content)
            temp_html_path = Path(temp_html.name)

        page = await self.setup_browser_page()

        try:
            html_url = temp_html_path.resolve().as_uri()
            await page.goto(html_url, wait_until='domcontentloaded')
            await self.wait_for_mathjax(page)

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                temp_pdf_path = Path(temp_pdf.name)
                await self._generate_pdf(page, temp_pdf_path)
                result = self._check_pdf_page_count(temp_pdf_path)
                temp_pdf_path.unlink()

            return result
        finally:
            await page.close()
            temp_html_path.unlink()


    async def generate_pdf_from_html(self, html_path: Path, pdf_path: Path) -> None:
        """Converts HTML to PDF using existing browser instance."""
        page = await self.setup_browser_page()
        
        try:
            html_url = html_path.resolve().as_uri()
            await page.goto(html_url, wait_until='domcontentloaded')
            await self.wait_for_mathjax(page)

            await self._generate_pdf(page, pdf_path)
        finally:
            await page.close()
    
    @staticmethod
    async def wait_for_mathjax(page: Page) -> None:
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
    def _check_pdf_page_count(pdf_path: Path) -> bool:
        """Check if PDF has exactly one page."""
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            page_count = len(pdf_reader.pages)
            is_single_page = page_count == 1
            logger.debug(f"PDF has {page_count} page(s), Single page: {is_single_page}")
            return is_single_page