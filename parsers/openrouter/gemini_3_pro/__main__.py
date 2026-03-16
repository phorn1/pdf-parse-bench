"""CLI entry point for Gemini 3 Pro parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openrouter.base_openrouter_parser import BaseOpenRouterParser


class Gemini3ProParser(BaseOpenRouterParser):
    """Parser using Google Gemini 3 Pro model."""

    model = "google/gemini-3-pro-preview"

    @classmethod
    def display_name(cls) -> str:
        return "Gemini 3 Pro"


if __name__ == "__main__":
    run_cli(parser=Gemini3ProParser())
