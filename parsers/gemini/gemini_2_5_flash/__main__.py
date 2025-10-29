from pdf_parse_bench.pipeline import run_cli
from parsers.gemini.base_gemini_parser import BaseGeminiParser


class GeminiFlashParser(BaseGeminiParser):
    """Parser using Google Gemini 2.5 Flash model."""

    model = "gemini-2.5-flash"

    @classmethod
    def parser_name(cls) -> str:
        """Return parser name identifier."""
        return "gemini_2_5_flash"


if __name__ == "__main__":
    run_cli(parser=GeminiFlashParser())