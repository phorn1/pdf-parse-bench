"""CLI entry point for Gemini 2.5 Pro parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openrouter.base_openrouter_parser import BaseOpenRouterParser


class Gemini25ProParser(BaseOpenRouterParser):
    """Parser using Google Gemini 2.5 Pro model."""

    model = "google/gemini-2.5-pro"

    @classmethod
    def display_name(cls) -> str:
        return "Gemini 2.5 Pro"


if __name__ == "__main__":
    run_cli(parser=Gemini25ProParser())
