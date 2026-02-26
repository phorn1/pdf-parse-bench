from pathlib import Path

from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser


class PaddleOCRVLParser(PDFParser):
    """PDF parser using PaddleOCR-VL (PaddlePaddle OCR Vision-Language)."""

    def __init__(self):
        """Initialize PaddleOCR-VL parser."""
        super().__init__()
        self.pipeline = None

    def _load_model(self):
        """Load PaddleOCR-VL model using official pipeline."""
        if self.pipeline is not None:
            return

        from paddleocr import PaddleOCRVL

        self.pipeline = PaddleOCRVL()

    @classmethod
    def display_name(cls) -> str:
        return "PaddleOCR-VL"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using PaddleOCR-VL."""
        self._load_model()

        # PaddleOCR-VL handles PDFs directly — no image conversion needed
        result = next(iter(self.pipeline.predict(str(pdf_path))))
        markdown = result.markdown['markdown_texts']

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=PaddleOCRVLParser())