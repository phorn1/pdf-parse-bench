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
    def display_name(cls) -> str:
        """Return human-readable parser name (e.g., 'DeepSeek-OCR', 'Gemini 2.5 Flash')."""
        pass

    @classmethod
    def parser_id(cls) -> str:
        """Return unique parser identifier extracted from module path."""
        import inspect

        # Get file path of the class definition
        file_path = Path(inspect.getfile(cls))
        # Return parent directory name (e.g., "mistral" from "parsers/mistral/__main__.py")
        return file_path.parent.name

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