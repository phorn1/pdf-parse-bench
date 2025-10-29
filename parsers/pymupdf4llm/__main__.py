"""CLI entry point for PyMuPDF parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
import pymupdf4llm
from pathlib import Path
from pdf_parse_bench.utilities import PDFParser


class PyMuPDFParser(PDFParser):
    """PDF parser using PyMuPDF4LLM."""

    @classmethod
    def parser_name(cls) -> str:
        """Return parser name identifier."""
        return "PyMuPDF"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using PyMuPDF4LLM.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """
        markdown = pymupdf4llm.to_markdown(str(pdf_path))
        self._write_output(markdown, output_path)
        return markdown

if __name__ == "__main__":
    run_cli(parser=PyMuPDFParser())