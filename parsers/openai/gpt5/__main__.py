"""CLI entry point for GPT-5 parser benchmark."""

from pdf_benchmark.pipeline import run_cli
from parsers.openai.base_openai_parser import BaseOpenAIParser


class GPT5Parser(BaseOpenAIParser):
    """Parser using OpenAI GPT-5 model."""

    model = "gpt-5"

    @classmethod
    def parser_name(cls) -> str:
        """Return parser name identifier."""
        return "GPT-5"


if __name__ == "__main__":
    run_cli(parser=GPT5Parser())