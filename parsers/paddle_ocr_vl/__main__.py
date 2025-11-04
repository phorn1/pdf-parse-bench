import tempfile
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
    def parser_name(cls) -> str:
        """Return parser name identifier."""
        return "paddle_ocr_vl"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using PaddleOCR-VL."""
        self._load_model()

        # ========== PDF TO IMAGE ==========
        import fitz

        doc = fitz.open(pdf_path)
        page = doc[0]  # Single page only
        pix = page.get_pixmap(dpi=300)

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        pix.save(temp_path)
        doc.close()

        # ========== RUN OCR ==========
        try:
            # predict() returns a list of Result objects
            results = self.pipeline.predict(str(temp_path))
            res = results[0]  # Get first result (single image)
            # Extract markdown text from the result's markdown attribute
            markdown = res.markdown['markdown_texts']
        finally:
            temp_path.unlink()

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=PaddleOCRVLParser())