"""CLI entry point for PyPDF parser benchmark."""

from pathlib import Path

from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser


class PyPDFParser(PDFParser):
    """PDF parser using PyPDF."""

    @classmethod
    def display_name(cls) -> str:
        return "PyPDF"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using PyPDF.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """

        from pypdf import PdfReader

        pdf = PdfReader(str(pdf_path))
        pages = pdf.pages
        markdown = "\n\n".join([page.extract_text(0) for page in pages])

        self._write_output(markdown, output_path)
        return markdown

if __name__ == "__main__":
    run_cli(parser=PyPDFParser())