"""CLI entry point for Gemini 2.5 Flash parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openrouter.base_openrouter_parser import BaseOpenRouterParser


class Gemini25FlashParser(BaseOpenRouterParser):
    """Parser using Google Gemini 2.5 Flash model."""

    model = "google/gemini-2.5-flash"

    @classmethod
    def display_name(cls) -> str:
        return "Gemini 2.5 Flash"


if __name__ == "__main__":
    run_cli(parser=Gemini25FlashParser())
