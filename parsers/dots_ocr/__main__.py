import base64
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser


class DOTSOCRParser(PDFParser):
    """PDF parser using DOTS OCR (Rednote HiLab) via vLLM."""

    def __init__(self):
        """Initialize DOTS OCR parser with vLLM client."""
        super().__init__()

        self.vllm_url = "http://localhost:8000/v1"
        self.model_name = "rednote-hilab/dots.ocr"

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

    def _get_formula_in_markdown(self, text: str) -> str:
        """Format formula text into markdown (from original DOTS OCR format_transformer.py)."""
        text = text.strip()

        # Check if it's already enclosed in $$
        if text.startswith('$$') and text.endswith('$$'):
            text_new = text[2:-2].strip()
            if '$' not in text_new:
                return f"$$\n{text_new}\n$$"
            else:
                return text

        # Handle \[...\] format, convert to $$...$$
        if text.startswith('\\[') and text.endswith('\\]'):
            inner_content = text[2:-2].strip()
            return f"$$\n{inner_content}\n$$"

        # Handle inline formulas ($...$)
        if text.startswith('$') and text.endswith('$') and text.count('$') == 2:
            return text  # Keep inline formulas as is

        # If no special formatting, wrap in $$ block
        return f"$$\n{text}\n$$"

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace (from original DOTS OCR format_transformer.py)."""
        if not text:
            return ""

        text = text.strip()

        # Remove backtick wrapping from formulas
        if text.startswith('`$') and text.endswith('$`'):
            text = text[1:-1]

        return text

    def _layout_json_to_markdown(self, layout_data: list[dict]) -> str:
        """Convert layout JSON to markdown (based on original DOTS OCR layoutjson2md)."""
        text_items = []

        for element in layout_data:
            if not isinstance(element, dict):
                continue

            category = element.get("category", "")
            text = element.get("text", "")

            # Skip page headers and footers
            if category in ['Page-header', 'Page-footer']:
                continue

            # Skip pictures (no image content in markdown)
            if category == 'Picture':
                continue

            # Format formulas specially
            if category == 'Formula':
                text_items.append(self._get_formula_in_markdown(text))
            else:
                text = self._clean_text(text)
                if text:
                    text_items.append(text)

        return '\n\n'.join(text_items)

    @classmethod
    def parser_name(cls) -> str:
        """Return parser name identifier."""
        return "dots_ocr"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using DOTS OCR via vLLM."""
        # ========== PDF TO IMAGE ==========
        import fitz
        from PIL import Image

        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=200)  # DOTS OCR recommends 200 dpi

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        pix.save(temp_path)
        doc.close()

        # ========== RUN OCR WITH VLLM ==========
        try:
            image = Image.open(temp_path)
            base64_image = self._encode_image_to_base64(image)

            prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

            # Call vLLM API using OpenAI client
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.1,
                top_p=0.9,
                max_tokens=24000,
            )

            json_output = response.choices[0].message.content
            layout_data = json.loads(json_output)
            markdown = self._layout_json_to_markdown(layout_data)

        finally:
            temp_path.unlink()

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=DOTSOCRParser())