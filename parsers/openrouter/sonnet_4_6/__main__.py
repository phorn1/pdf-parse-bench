"""CLI entry point for Claude Sonnet 4.6 parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openrouter.base_openrouter_parser import BaseOpenRouterParser


class Sonnet46Parser(BaseOpenRouterParser):
    """Parser using Anthropic Claude Sonnet 4.6 model."""

    model = "anthropic/claude-sonnet-4.6"

    @classmethod
    def display_name(cls) -> str:
        return "Claude Sonnet 4.6"


if __name__ == "__main__":
    run_cli(parser=Sonnet46Parser())
