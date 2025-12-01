from pdf_parse_bench.pipeline import run_cli
from parsers.gemini.base_gemini_parser import BaseGeminiParser


class Gemini3ProParser(BaseGeminiParser):
    """Parser using Google Gemini 3 Pro model."""

    model = "gemini-3-pro-preview"

    @classmethod
    def display_name(cls) -> str:
        return "Gemini 3 Pro"


if __name__ == "__main__":
    run_cli(parser=Gemini3ProParser())