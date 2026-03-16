"""CLI entry point for GPT-5.2 parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openrouter.base_openrouter_parser import BaseOpenRouterParser


class GPT52Parser(BaseOpenRouterParser):
    """Parser using OpenAI GPT-5.2 model."""

    model = "openai/gpt-5.2"

    @classmethod
    def display_name(cls) -> str:
        return "GPT-5.2"


if __name__ == "__main__":
    run_cli(parser=GPT52Parser())
