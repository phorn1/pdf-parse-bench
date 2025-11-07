from pdf_parse_bench.pipeline import run_cli
from parsers.gemini.base_gemini_parser import BaseGeminiParser


class GeminiProParser(BaseGeminiParser):
    """Parser using Google Gemini 2.5 Pro model."""

    model = "gemini-2.5-pro"

    @classmethod
    def display_name(cls) -> str:
        return "Gemini 2.5 Pro"


if __name__ == "__main__":
    run_cli(parser=GeminiProParser())