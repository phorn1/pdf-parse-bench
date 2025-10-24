"""OpenAI multimodal parser - uses GPT-5 models for PDF parsing."""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from pdf_benchmark.utilities import PDFParser
from pdf_benchmark.utilities.vlm_prompt import PDF_TO_MARKDOWN_PROMPT

# Load environment variables
load_dotenv()


class BaseOpenAIParser(PDFParser):
    """Base parser for OpenAI multimodal models."""

    # Subclasses should set this
    model: str = None

    def __init__(self):
        """Initialize OpenAI parser."""
        super().__init__()
        if self.model is None:
            raise ValueError(f"{self.__class__.__name__} must define a 'model' class attribute")

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using OpenAI's responses API.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        # Read and encode PDF
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        base64_pdf = base64.b64encode(pdf_data).decode("utf-8")

        # Use the responses.create API
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": pdf_path.name,
                            "file_data": f"data:application/pdf;base64,{base64_pdf}",
                        },
                        {
                            "type": "input_text",
                            "text": PDF_TO_MARKDOWN_PROMPT,
                        },
                    ]
                }
            ]
        )

        # Extract markdown from response
        markdown = response.output_text

        # Write output
        self._write_output(markdown, output_path)
        return markdown