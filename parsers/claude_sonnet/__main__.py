"""CLI entry point for Claude Sonnet parser benchmark (direct Anthropic API)."""

import os
import base64
from pathlib import Path

from dotenv import load_dotenv

from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser
from pdf_parse_bench.utilities.vlm_prompt import PDF_TO_MARKDOWN_PROMPT

load_dotenv()


class ClaudeSonnetParser(PDFParser):
    """Parser using Claude Sonnet 4.6 via the Anthropic API."""

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    @classmethod
    def display_name(cls) -> str:
        return "Claude Sonnet 4.6"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16384,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64_pdf,
                        },
                    },
                    {"type": "text", "text": PDF_TO_MARKDOWN_PROMPT},
                ],
            }],
        )

        markdown = message.content[0].text
        if not markdown:
            raise ValueError(f"Claude Sonnet returned empty content for {pdf_path.name}")

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=ClaudeSonnetParser())