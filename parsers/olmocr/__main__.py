"""CLI entry point for OLMo-OCR parser benchmark."""

import base64
import tempfile
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser


class OLMoOCRParser(PDFParser):
    """PDF parser using OLMo-OCR (AllenAI) via vLLM."""

    def __init__(self):
        """Initialize OLMo-OCR parser with vLLM client."""
        super().__init__()

        self.vllm_url = "http://localhost:8000/v1"
        self.model_name = "olmocr"  # Name as served by vLLM

        # Initialize OpenAI client for vLLM
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require an API key
            base_url=self.vllm_url,
        )

    def _encode_image_to_base64(self, image) -> str:
        """Encode PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @classmethod
    def display_name(cls) -> str:
        return "olmOCR-2-7B-1025-FP8"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using OLMo-OCR via vLLM."""
        # ========== PDF TO IMAGE ==========
        import fitz
        from PIL import Image

        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=300)

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        pix.save(temp_path)
        doc.close()

        # ========== RUN OCR WITH VLLM ==========
        try:
            image = Image.open(temp_path)
            base64_image = self._encode_image_to_base64(image)

            prompt = """Attached is one page of a document that you must process. Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.
If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)
Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."""

            # Call vLLM API using OpenAI client
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=8000,
            )

            markdown = response.choices[0].message.content

        finally:
            temp_path.unlink()

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=OLMoOCRParser())