"""LaTeX-based PDF generation module."""

from .style_config import LaTeXConfig
from .pipeline import ParallelPDFGenerator, PDFJob, SinglePagePDFGenerator

__all__ = [
    'LaTeXConfig',
    'PDFJob',
    'ParallelPDFGenerator',
    'SinglePagePDFGenerator',
]