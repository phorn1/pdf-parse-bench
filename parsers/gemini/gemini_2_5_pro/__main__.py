from pdf_benchmark.pipeline import run_cli
from parsers.gemini.base_gemini_parser import BaseGeminiParser


class GeminiProParser(BaseGeminiParser):
    """Parser using Google Gemini 2.5 Pro model."""

    model = "gemini-2.5-pro"

    @classmethod
    def parser_name(cls) -> str:
        """Return parser name identifier."""
        return "gemini_2_5_pro"


if __name__ == "__main__":
    run_cli(parser=GeminiProParser())