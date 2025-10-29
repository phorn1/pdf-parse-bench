"""CLI entry point for GPT-5-nano parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openai.base_openai_parser import BaseOpenAIParser


class GPT5NanoParser(BaseOpenAIParser):
    """Parser using OpenAI GPT-5-nano model."""

    model = "gpt-5-nano"

    @classmethod
    def parser_name(cls) -> str:
        """Return parser name identifier."""
        return "gpt_5_nano"


if __name__ == "__main__":
    run_cli(parser=GPT5NanoParser())