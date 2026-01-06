"""LaTeX-based PDF generation module."""

from .latex_config import LaTeXConfig
from .pipeline import ParallelPDFGenerator, PDFJob, SinglePagePDFGenerator

__all__ = [
    'LaTeXConfig',
    'PDFJob',
    'ParallelPDFGenerator',
    'SinglePagePDFGenerator',
]