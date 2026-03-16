"""Base parser for multimodal models via OpenRouter."""

import os
import base64
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from pdf_parse_bench.utilities import PDFParser
from pdf_parse_bench.utilities.vlm_prompt import PDF_TO_MARKDOWN_PROMPT

load_dotenv()


class BaseOpenRouterParser(PDFParser):
    """Base parser for multimodal models via OpenRouter.

    Subclasses set `model` and optionally `input_mode`:
      - "pdf"   → uploads the PDF directly (type: file)
      - "image" → converts first page to PNG, uploads as image_url
    """

    model: str = None
    input_mode: str = "pdf"  # "pdf" or "image"

    def __init__(self):
        super().__init__()
        if self.model is None:
            raise ValueError(f"{self.__class__.__name__} must define a 'model' class attribute")

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

    def _build_attachment(self, pdf_path: Path) -> dict:
        """Build the content attachment based on input_mode."""
        if self.input_mode == "pdf":
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            return {
                "type": "file",
                "file": {
                    "filename": pdf_path.name,
                    "file_data": f"data:application/pdf;base64,{base64_pdf}",
                },
            }

        # input_mode == "image"
        import fitz

        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=300)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = Path(tmp.name)
        pix.save(tmp_path)
        doc.close()

        try:
            with open(tmp_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        finally:
            tmp_path.unlink()

        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
        }

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse PDF to markdown using a multimodal model via OpenRouter."""
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        attachment = self._build_attachment(pdf_path)

        completion = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PDF_TO_MARKDOWN_PROMPT},
                    attachment,
                ],
            }],
        )

        markdown = completion.choices[0].message.content
        if not markdown:
            raise ValueError(f"Model {self.model} returned empty content for {pdf_path.name}")

        self._write_output(markdown, output_path)
        return markdown
