"""CLI entry point for Nanonets-OCR-s parser benchmark."""

import tempfile
from pathlib import Path

from pdf_benchmark.pipeline import run_cli
from pdf_benchmark.utilities import PDFParser


class NanonetsOCRSParser(PDFParser):
    """PDF parser using Nanonets-OCR-s with transformers."""

    def __init__(self):
        """Initialize Nanonets-OCR-s parser."""
        super().__init__()
        self.model = None
        self.processor = None

    def _load_model(self):
        """Load Nanonets-OCR-s model using transformers."""
        if self.model is not None:
            return

        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_path = "nanonets/Nanonets-OCR-s"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_path)

    @classmethod
    def parser_name(cls) -> str:
        """Return parser name identifier."""
        return "Nanonets-OCR-s"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using Nanonets-OCR-s."""
        self._load_model()

        # ========== PDF TO IMAGE ==========
        import fitz
        from PIL import Image

        doc = fitz.open(pdf_path)
        page = doc[0]  # Single page only
        pix = page.get_pixmap(dpi=300)

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        pix.save(temp_path)
        doc.close()

        # ========== RUN OCR WITH TRANSFORMERS ==========
        try:
            image = Image.open(temp_path)

            # Using exact prompt from model card
            prompt = "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{temp_path}"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            output_ids = self.model.generate(
                **inputs, max_new_tokens=15000, do_sample=False
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            markdown = output_text[0]

        finally:
            temp_path.unlink()

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=NanonetsOCRSParser())