from pypdf import PdfReader
from pathlib import Path

from .core import PDFParser, parser_registry


@parser_registry()
class PyPDFParser(PDFParser):
    """PDF parser using PyPDF."""
    
    @classmethod
    def parser_name(cls) -> str:
        return "pypdf"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using PyPDF.
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file
            
        Returns:
            str: Generated markdown content
        """
        
        pdf = PdfReader(str(pdf_path))
        pages = pdf.pages
        markdown = "\n\n".join([page.extract_text(0) for page in pages])
        
        self._write_output(markdown, output_path)
        return markdown


