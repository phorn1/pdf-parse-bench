import os
from pathlib import Path
from dotenv import load_dotenv
from pdf_benchmark.utilities import PDFParser
from pdf_benchmark.utilities.vlm_prompt import PDF_TO_MARKDOWN_PROMPT

# Load environment variables
load_dotenv()


class BaseGeminiParser(PDFParser):
    """Base parser for Gemini multimodal models."""

    # Subclasses should set this
    model: str = None

    def __init__(self):
        """Initialize Gemini parser."""
        super().__init__()
        if self.model is None:
            raise ValueError(f"{self.__class__.__name__} must define a 'model' class attribute")

        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using Gemini API.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        with open(pdf_path, "rb") as file:
            pdf_bytes = file.read()

        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
                PDF_TO_MARKDOWN_PROMPT
            ]
        )

        # Write output
        self._write_output(response.text, output_path)
        return response.text