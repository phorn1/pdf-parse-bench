import os
from pathlib import Path
from dotenv import load_dotenv
from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser

# Load environment variables
load_dotenv()


class MathpixParser(PDFParser):
    """PDF parser using Mathpix OCR."""

    def __init__(self):
        super().__init__()
        self.app_id = os.getenv("MATHPIX_APP_ID")
        self.app_key = os.getenv("MATHPIX_APP_KEY")

        if not self.app_id or not self.app_key:
            raise ValueError("MATHPIX_APP_ID and MATHPIX_APP_KEY environment variables are required")

    @classmethod
    def parser_name(cls) -> str:
        return "mathpix"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using Mathpix OCR.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """
        from mpxpy.mathpix_client import MathpixClient

        client = MathpixClient(
            app_id=self.app_id,
            app_key=self.app_key
        )

        # Process the PDF file
        pdf = client.pdf_new(
            file_path=str(pdf_path),
            convert_to_md=True,
        )

        # Wait for processing to complete
        pdf.wait_until_complete(timeout=60)

        # Get the markdown content
        markdown_content = pdf.to_md_text()

        self._write_output(markdown_content, output_path)
        return markdown_content


if __name__ == "__main__":
    run_cli(parser=MathpixParser())