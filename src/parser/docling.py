from pathlib import Path
import base64
import requests
from pdf2image import convert_from_path
from io import BytesIO

from .core import PDFParser, parser_registry


@parser_registry()
class DoclingParser(PDFParser):
    """PDF parser using SmolDocling API."""

    @classmethod
    def parser_name(cls) -> str:
        return "docling"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using SmolDocling API.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """
        # Convert PDF to image (single page)
        images = convert_from_path(str(pdf_path), dpi=300)
        image = images[0]  # Take first (and only) page

        # Convert image to base64
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Send to SmolDocling API
        response = requests.post(
            "http://localhost:8000/ocr/smol",  # Adjust URL as needed
            json={"image_base64": image_base64}
        )

        if response.status_code == 200:
            markdown_content = response.json()["result"]
        else:
            markdown_content = f"Error: {response.text}"

        self._write_output(markdown_content, output_path)
        return markdown_content