import pymupdf4llm
from pathlib import Path

from .core import PDFParser, parser_registry


@parser_registry()
class PyMuPDFParser(PDFParser):
    """PDF parser using PyMuPDF4LLM."""
    
    @classmethod
    def parser_name(cls) -> str:
        return "pymupdf"
    
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


