"""CLI entry point for Qwen3-VL parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openrouter.base_openrouter_parser import BaseOpenRouterParser


class Qwen3VLParser(BaseOpenRouterParser):
    """Parser using Qwen3-VL-235B-A22B-Instruct (image-only model)."""

    model = "qwen/qwen3-vl-235b-a22b-instruct"
    input_mode = "image"

    @classmethod
    def display_name(cls) -> str:
        return "Qwen3-VL-235B-A22B-Instruct"


if __name__ == "__main__":
    run_cli(parser=Qwen3VLParser())