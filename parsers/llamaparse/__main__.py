import os
from pathlib import Path
from dotenv import load_dotenv
from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser

# Load environment variables
load_dotenv()


class LlamaParseParser(PDFParser):
    """PDF parser using LlamaParse."""

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("LLAMACLOUD_API_KEY")

        if not self.api_key:
            raise ValueError("LLAMACLOUD_API_KEY environment variable is required")

    @classmethod
    def display_name(cls) -> str:
        return "LlamaParse"
    

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using LlamaParse.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """
        from llama_cloud_services import LlamaParse

        parser = LlamaParse(
            api_key=self.api_key,
            verbose=True,
            premium_mode=True
        )

        result = parser.parse(str(pdf_path))

        documents = result.get_markdown_documents(split_by_page=False)
        markdown = documents[0].text
        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=LlamaParseParser())