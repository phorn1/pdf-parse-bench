"""CLI entry point for Qwen3-VL parser benchmark."""

import base64
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser

# Load environment variables
load_dotenv()


class Qwen3VLParser(PDFParser):
    """PDF parser using Qwen3-VL-235B-A22B-Instruct via OpenRouter."""

    def __init__(self):
        """Initialize Qwen3-VL parser."""
        super().__init__()
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "qwen/qwen3-vl-235b-a22b-instruct"

        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

    @classmethod
    def display_name(cls) -> str:
        return "Qwen3-VL-235B-A22B-Instruct"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using Qwen3-VL via OpenRouter."""
        from openai import OpenAI

        # ========== PDF TO IMAGE ==========
        import fitz

        doc = fitz.open(pdf_path)
        page = doc[0]  # Single page only
        pix = page.get_pixmap(dpi=300)

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        pix.save(temp_path)
        doc.close()

        # ========== CONVERT IMAGE TO BASE64 ==========
        try:
            with open(temp_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # ========== CALL OPENROUTER API ==========
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
            )

            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Convert the document to markdown."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )

            markdown = completion.choices[0].message.content

        finally:
            temp_path.unlink()

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=Qwen3VLParser())