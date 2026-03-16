"""CLI entry point for Claude Opus 4.6 parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openrouter.base_openrouter_parser import BaseOpenRouterParser


class Opus46Parser(BaseOpenRouterParser):
    """Parser using Anthropic Claude Opus 4.6 model."""

    model = "anthropic/claude-opus-4.6"

    @classmethod
    def display_name(cls) -> str:
        return "Claude Opus 4.6"


if __name__ == "__main__":
    run_cli(parser=Opus46Parser())
