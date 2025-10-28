"""CLI entry point for GOT-OCR2.0 parser benchmark."""

import tempfile
from pathlib import Path

from pdf_benchmark.pipeline import run_cli
from pdf_benchmark.utilities import PDFParser


class GOTOCR2Parser(PDFParser):
    """PDF parser using GOT-OCR2.0 (General OCR Theory)."""

    def __init__(self, ocr_type: str = "format", device_map: str = "cuda"):
        """
        Initialize GOT-OCR2.0 parser.

        Args:
            ocr_type: 'ocr' for plain text, 'format' for formatted output
            device_map: 'cuda' for GPU, 'cpu' for CPU
        """
        super().__init__()
        self.ocr_type = ocr_type
        self.device_map = device_map
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """Load GOT-OCR2.0 model and tokenizer."""
        if self.model is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            'ucaslcl/GOT-OCR2_0',
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            'ucaslcl/GOT-OCR2_0',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self.device_map,
            use_safetensors=True,
            pad_token_id=self.tokenizer.eos_token_id
        ).eval()

        if self.device_map == "cuda":
            self.model = self.model.cuda()

    @classmethod
    def parser_name(cls) -> str:
        """Return parser name identifier."""
        return "got_ocr2"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse single-page PDF to markdown using GOT-OCR2.0.

        Args:
            pdf_path: Path to input PDF file (single page)
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """
        self._load_model()

        # ========== PDF TO IMAGE ==========
        import fitz

        doc = fitz.open(pdf_path)
        page = doc[0]  # Single page only
        pix = page.get_pixmap(dpi=300)

        # Save pixmap directly to temp PNG file
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        pix.save(temp_path)
        doc.close()

        # ========== RUN OCR ==========
        try:
            markdown = self.model.chat(
                self.tokenizer,
                str(temp_path),
                ocr_type=self.ocr_type
            )
        finally:
            temp_path.unlink()

        # ========== WRITE OUTPUT ==========
        self._write_output(markdown, output_path)
        return markdown

if __name__ == "__main__":
    run_cli(parser=GOTOCR2Parser())