from pathlib import Path

from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser


class PPStructureV3Parser(PDFParser):
    """PDF parser using PP-StructureV3 (PaddlePaddle Structure Recognition)."""

    def __init__(self):
        """Initialize PP-StructureV3 parser."""
        super().__init__()
        self.pipeline = None

    def _load_model(self):
        """Load PP-StructureV3 pipeline."""
        if self.pipeline is not None:
            return

        from paddleocr import PPStructureV3

        self.pipeline = PPStructureV3()

    @classmethod
    def parser_name(cls) -> str:
        """Return parser name identifier."""
        return "pp_structure_v3_3"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using PP-StructureV3."""
        self._load_model()

        # ========== RUN PP-STRUCTUREV3 ==========
        # PP-StructureV3 can process PDFs directly
        results = self.pipeline.predict(input=str(pdf_path))

        # Extract markdown from first page (single-page PDFs)
        res = results[0]
        md_info = res.markdown

        # Extract markdown text from the result
        if isinstance(md_info, dict) and "markdown_texts" in md_info:
            markdown = md_info["markdown_texts"]
        elif hasattr(md_info, "markdown_texts"):
            markdown = md_info.markdown_texts
        else:
            # Fallback: use concatenate_markdown_pages for consistency
            markdown = self.pipeline.concatenate_markdown_pages([md_info])

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=PPStructureV3Parser())