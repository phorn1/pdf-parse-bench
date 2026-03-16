"""CLI entry point for GPT-5-nano parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openrouter.base_openrouter_parser import BaseOpenRouterParser


class GPT5NanoParser(BaseOpenRouterParser):
    """Parser using OpenAI GPT-5-nano model."""

    model = "openai/gpt-5-nano"

    @classmethod
    def display_name(cls) -> str:
        return "GPT-5 nano"


if __name__ == "__main__":
    run_cli(parser=GPT5NanoParser())
