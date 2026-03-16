"""CLI entry point for GPT-5-mini parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openrouter.base_openrouter_parser import BaseOpenRouterParser


class GPT5MiniParser(BaseOpenRouterParser):
    """Parser using OpenAI GPT-5-mini model."""

    model = "openai/gpt-5-mini"

    @classmethod
    def display_name(cls) -> str:
        return "GPT-5 mini"


if __name__ == "__main__":
    run_cli(parser=GPT5MiniParser())
