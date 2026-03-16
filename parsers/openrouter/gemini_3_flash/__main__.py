"""CLI entry point for Gemini 3 Flash parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openrouter.base_openrouter_parser import BaseOpenRouterParser


class Gemini3FlashParser(BaseOpenRouterParser):
    """Parser using Google Gemini 3 Flash model."""

    model = "google/gemini-3-flash-preview"

    @classmethod
    def display_name(cls) -> str:
        return "Gemini 3 Flash"


if __name__ == "__main__":
    run_cli(parser=Gemini3FlashParser())
