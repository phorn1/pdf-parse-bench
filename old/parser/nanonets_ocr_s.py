import base64
import requests
from pathlib import Path
import pdf2image

from .core import PDFParser, parser_registry


@parser_registry()
class NanonetsParser(PDFParser):
    """PDF parser using Nanonets OCR."""

    def __init__(self):
        super().__init__()
        self.base_url = "http://localhost:8000"

    @classmethod
    def parser_name(cls) -> str:
        return "nanonets_ocr_s"

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _ocr_page_with_nanonets(self, img_base64: str) -> str:
        """Process a single page image with Nanonets OCR."""
        try:
            response = requests.post(
                f"{self.base_url}/ocr/nanonets",
                json={"image_base64": img_base64},
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minutes timeout for OCR processing
            )
            response.raise_for_status()
            return response.json()["result"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Nanonets API request failed: {e}")
        except KeyError:
            raise RuntimeError("Unexpected response format from Nanonets API")

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using Nanonets OCR.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """
        from io import BytesIO

        # Convert PDF to single page image
        image = pdf2image.convert_from_path(pdf_path)[0]

        # Convert image to base64
        img_buffer = BytesIO()
        image.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        # Process with Nanonets
        markdown_content = self._ocr_page_with_nanonets(img_base64)

        self._write_output(markdown_content, output_path)
        return markdown_content
