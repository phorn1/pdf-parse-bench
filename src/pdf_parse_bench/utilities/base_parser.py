"""Core parser base classes."""

from abc import ABC, abstractmethod
from pathlib import Path
import logging


class PDFParser(ABC):
    """Abstract base class for all PDF parsers."""

    def __init__(self):
        """Initialize parser."""
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    @classmethod
    @abstractmethod
    def parser_name(self) -> str:
        """Return parser name identifier."""
        pass

    @abstractmethod
    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file

        Returns:
            str: Generated markdown content
        """
        pass

    def _write_output(self, content: str, output_path: Path) -> None:
        """Write content to output file with UTF-8 encoding."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)