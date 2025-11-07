from pdf_parse_bench.pipeline import run_cli
from parsers.gemini.base_gemini_parser import BaseGeminiParser


class GeminiFlashParser(BaseGeminiParser):
    """Parser using Google Gemini 2.5 Flash model."""

    model = "gemini-2.5-flash"

    @classmethod
    def display_name(cls) -> str:
        return "Gemini 2.5 Flash"


if __name__ == "__main__":
    run_cli(parser=GeminiFlashParser())