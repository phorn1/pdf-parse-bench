from pathlib import Path

from .core import PDFParser, parser_registry


@parser_registry()
class UnstructuredParser(PDFParser):
    """PDF parser using Unstructured library."""
    
    @classmethod
    def parser_name(cls) -> str:
        return "unstructured"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using Unstructured.
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file
            
        Returns:
            str: Generated markdown content
        """
        
        from unstructured.partition.auto import partition
        
        elements = partition(
            filename=str(pdf_path),
            strategy="hi_res",
            # extract_images_in_pdf=True,
            # infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            # extract_image_block_to_payload=True,
        )
        text = "\n\n".join([str(el) for el in elements])
        
        self._write_output(text, output_path)
        return text



