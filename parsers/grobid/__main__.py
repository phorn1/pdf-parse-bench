"""CLI entry point for GROBID parser benchmark."""

import tempfile
from pathlib import Path

from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser


class GROBIDParser(PDFParser):
    """PDF parser using GROBID (GeneRation Of BIbliographic Data)."""

    def __init__(self):
        """Initialize GROBID parser with client configuration."""
        super().__init__()
        self.grobid_server = "http://localhost:8070"

    @classmethod
    def display_name(cls) -> str:
        return "GROBID"

    def _convert_tei_to_markdown(self, tei_xml: str) -> str:
        """
        Convert GROBID TEI XML output to markdown format.

        Args:
            tei_xml: TEI XML string from GROBID

        Returns:
            str: Markdown formatted text
        """
        from grobid_tei_xml import parse_document_xml

        # Parse TEI XML to extract structured content
        doc = parse_document_xml(tei_xml)

        markdown_parts = []

        # ========== ABSTRACT ==========
        if doc.abstract:
            markdown_parts.append(doc.abstract)

        # ========== FULL TEXT ==========
        if doc.body:
            markdown_parts.append(doc.body)

        return "\n\n".join(markdown_parts)

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using GROBID.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """
        import requests

        # ========== CALL GROBID API ==========
        with open(pdf_path, 'rb') as pdf_file:
            files = {
                'input': pdf_file,
            }
            params = {
                'consolidateHeader': '1',
                'consolidateCitations': '0',  # Faster without citation consolidation
                'includeRawCitations': '0',
                'includeRawAffiliations': '0',
                'teiCoordinates': '0',
                'segmentSentences': '0',
            }

            response = requests.post(
                f"{self.grobid_server}/api/processFulltextDocument",
                files=files,
                params=params,
                timeout=300,  # 5 minutes timeout
            )
            response.raise_for_status()

            tei_xml = response.text

        # ========== CONVERT TEI TO MARKDOWN ==========
        markdown = self._convert_tei_to_markdown(tei_xml)

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=GROBIDParser())