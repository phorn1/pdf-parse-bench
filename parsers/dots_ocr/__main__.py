"""CLI entry point for DOTS OCR parser benchmark."""

import json
import tempfile
from pathlib import Path

from pdf_benchmark.pipeline import run_cli
from pdf_benchmark.utilities import PDFParser


class DOTSOCRParser(PDFParser):
    """PDF parser using DOTS OCR (Rednote HiLab)."""

    def __init__(self):
        """Initialize DOTS OCR parser."""
        super().__init__()
        self.model = None
        self.processor = None

    def _load_model(self):
        """Load DOTS OCR model using transformers."""
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = "rednote-hilab/dots.ocr"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

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
        """Parse single-page PDF to markdown using DOTS OCR."""
        self._load_model()

        # ========== PDF TO IMAGE ==========
        import fitz
        from PIL import Image

        doc = fitz.open(pdf_path)
        page = doc[0]  # Single page only
        pix = page.get_pixmap(dpi=200)  # DOTS OCR recommends 200 DPI

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        pix.save(temp_path)
        doc.close()

        # ========== RUN OCR WITH DOTS ==========
        try:
            from qwen_vl_utils import process_vision_info

            # Using the official prompt from the model card
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

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(temp_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to(self.model.device)

            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            json_output = output_text[0]

            # Parse JSON and convert to markdown
            layout_data = json.loads(json_output)
            markdown = self._layout_json_to_markdown(layout_data)

        finally:
            temp_path.unlink()

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=DOTSOCRParser())