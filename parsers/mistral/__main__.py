"""CLI entry point for Mistral parser benchmark."""

import base64
import os
from pathlib import Path
from dotenv import load_dotenv
from pdf_benchmark.pipeline import run_cli
from pdf_benchmark.utilities import PDFParser

# Load environment variables
load_dotenv()


class MistralParser(PDFParser):
    """PDF parser using Mistral OCR."""

    def __init__(self):
        super().__init__()
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.model = "mistral-ocr-latest"

        if not self.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")

    @classmethod
    def parser_name(cls) -> str:
        return "mistral"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using Mistral OCR.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """
        from mistralai import Mistral

        client = Mistral(api_key=self.mistral_api_key)

        with open(pdf_path, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')

        ocr_response = client.ocr.process(
            model=self.model,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            }
        )

        markdown_content = ""
        for page in ocr_response.pages:
            markdown_content += page.markdown

        self._write_output(markdown_content, output_path)
        return markdown_content


if __name__ == "__main__":
    run_cli(parser=MistralParser())