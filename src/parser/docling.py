from pathlib import Path
from docling.document_converter import DocumentConverter

from .core import PDFParser, parser_registry


@parser_registry()
class DoclingParser(PDFParser):
    """PDF parser using Docling."""
    
    @classmethod
    def parser_name(cls) -> str:
        return "docling"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using Docling.
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file
            
        Returns:
            str: Generated markdown content
        """
        from docling.datamodel import vlm_model_specs
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            VlmPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                ),
            }
        )

        result = converter.convert(str(pdf_path))
        markdown_content = result.document.export_to_markdown()
        
        self._write_output(markdown_content, output_path)
        return markdown_content