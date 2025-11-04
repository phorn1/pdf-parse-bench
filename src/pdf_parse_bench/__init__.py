"""PDF Parser Benchmark - A framework for evaluating PDF parsing quality."""

from .pipeline import Benchmark, run_benchmark
from .utilities.base_parser import PDFParser
from .utilities.data import get_benchmark_ground_truth_dir, get_benchmark_pdfs_dir

__all__ = [
    "Benchmark",
    "run_benchmark",
    "PDFParser",
    "get_benchmark_pdfs_dir",
    "get_benchmark_ground_truth_dir",
]