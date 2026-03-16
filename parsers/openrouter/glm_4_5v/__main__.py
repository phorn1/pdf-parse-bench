"""CLI entry point for GLM-4.5V parser benchmark."""

from pdf_parse_bench.pipeline import run_cli
from parsers.openrouter.base_openrouter_parser import BaseOpenRouterParser


class GLM45VParser(BaseOpenRouterParser):
    """Parser using GLM-4.5V (image-only model)."""

    model = "z-ai/glm-4.5v"
    input_mode = "image"

    @classmethod
    def display_name(cls) -> str:
        return "GLM-4.5V"


if __name__ == "__main__":
    run_cli(parser=GLM45VParser())