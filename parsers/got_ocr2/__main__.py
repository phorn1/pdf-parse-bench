"""CLI entry point for GOT-OCR2.0 parser benchmark."""

from pdf_benchmark.pipeline import run_cli
from parsers.got_ocr2.got_ocr2_parser import GOTOCR2Parser


if __name__ == "__main__":
    run_cli(parser=GOTOCR2Parser())