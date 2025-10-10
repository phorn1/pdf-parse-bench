import io
import json
import base64
import requests
from pathlib import Path
from PIL import Image
import fitz

from .core import PDFParser, parser_registry


# ========== ORIGINAL PROMPT FROM DOTS_OCR/UTILS/PROMPTS.PY ==========
PROMPT_LAYOUT_ALL_EN = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

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


@parser_registry()
class DotsOCRParser(PDFParser):
    """PDF parser using DotsOCR API."""

    def __init__(self):
        super().__init__()
        # Configuration from original DotsOCR project
        self.api_url = "http://localhost:8000/v1/chat/completions"

        self.model_name = "dotsocr-model"
        self.temperature = 0.1
        self.top_p = 0.9
        self.max_completion_tokens = 32768
        self.dpi = 200  # From original project default

    @classmethod
    def parser_name(cls) -> str:
        return "dots_ocr"

    def _pil_image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """Convert PIL image to base64 data URL (from original utils/image_utils.py)"""
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{base64_str}"

    def _fitz_doc_to_image(self, page, target_dpi: int = 200) -> Image.Image:
        """Convert fitz page to PIL image (from original dots_ocr/utils/doc_utils.py)"""
        mat = fitz.Matrix(target_dpi / 72, target_dpi / 72)
        pm = page.get_pixmap(matrix=mat, alpha=False)

        # Fallback to default DPI if image is too large (from original)
        if pm.width > 4500 or pm.height > 4500:
            mat = fitz.Matrix(72 / 72, 72 / 72)  # use fitz default dpi
            pm = page.get_pixmap(matrix=mat, alpha=False)

        image = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
        return image


    def _query_dots_ocr_api(self, image: Image.Image) -> dict:
        """Query DotsOCR API with exact original settings"""
        image_base64 = self._pil_image_to_base64(image)

        # Build request EXACTLY like original inference_with_vllm()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64}
                    },
                    {
                        "type": "text",
                        "text": f"<|img|><|imgpad|><|endofimg|>{PROMPT_LAYOUT_ALL_EN}"
                    }
                ]
            }
        ]

        payload = {
            "messages": messages,
            "model": self.model_name,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

        response = requests.post(
            self.api_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300
        )

        if response.status_code != 200:
            raise RuntimeError(f"DotsOCR API call failed: {response.status_code} - {response.text}")

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        layout_data = json.loads(content)
        return layout_data

    def _get_formula_in_markdown(self, text: str) -> str:
        """Format formula text into markdown (from original format_transformer.py)"""
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
        """Clean text by removing extra whitespace (from original format_transformer.py)"""
        if not text:
            return ""

        text = text.strip()

        # Remove backtick wrapping from formulas
        if text.startswith('`$') and text.endswith('$`'):
            text = text[1:-1]

        return text

    def _layout_json_to_markdown(self, layout_data, image: Image.Image = None) -> str:
        """Convert layout JSON to markdown (based on original layoutjson2md)"""
        # Handle case where layout_data is already a list (direct from API)
        layout_elements = layout_data
        text_items = []

        for element in layout_elements:
            if not isinstance(element, dict):
                continue

            category = element.get("category", "")
            text = element.get("text", "")

            # Skip page headers and footers (following original no_page_hf pattern)
            if category in ['Page-header', 'Page-footer']:
                continue

            if category == 'Picture':
                # Skip pictures as we don't have the image crop functionality here
                continue
            elif category == 'Formula':
                text_items.append(self._get_formula_in_markdown(text))
            else:
                text = self._clean_text(text)
                if text:
                    text_items.append(text)

        return '\n\n'.join(text_items)

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using DotsOCR API.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """

        # Convert PDF to image (expecting single page)
        with fitz.open(pdf_path) as doc:
            if doc.page_count != 1:
                self.logger.warning(f"PDF has {doc.page_count} pages, processing only the first page")

            page = doc[0]  # Get first (and expected only) page
            image = self._fitz_doc_to_image(page, target_dpi=self.dpi)


        try:
            layout_data = self._query_dots_ocr_api(image)
            final_markdown = self._layout_json_to_markdown(layout_data)

            if not final_markdown.strip():
                final_markdown = "*No content extracted from PDF*"

        except Exception as e:
            self.logger.error(f"Failed to process PDF: {e}")
            final_markdown = f"*Error processing PDF: {e}*"

        # Write output
        self._write_output(final_markdown, output_path)

        return final_markdown