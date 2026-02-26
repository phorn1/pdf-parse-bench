"""CLI entry point for Chandra OCR parser benchmark."""

import tempfile
from pathlib import Path

from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser


class ChandraParser(PDFParser):
    """PDF parser using Chandra OCR with HuggingFace transformers."""

    def __init__(self):
        """Initialize Chandra parser."""
        super().__init__()
        self.model = None

    def _load_model(self):
        """Load Chandra model using transformers."""
        if self.model is not None:
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_path = "datalab-to/chandra"

        self.model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        self.model.processor = AutoProcessor.from_pretrained(model_path)

    @classmethod
    def display_name(cls) -> str:
        return "Chandra"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using Chandra OCR."""
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

        # ========== RUN OCR WITH CHANDRA ==========
        try:
            from chandra.model.hf import generate_hf
            from chandra.model.schema import BatchInputItem
            from chandra.output import parse_markdown

            image = Image.open(temp_path).convert("RGB")

            batch = [BatchInputItem(image=image, prompt_type="ocr_layout")]
            result = generate_hf(batch, self.model)[0]
            markdown = parse_markdown(result.raw)

        finally:
            temp_path.unlink()

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=ChandraParser())
