"""Simplified registry system for parsers."""

from typing import Dict, Type, Callable, TypeVar
from pathlib import Path

from .base import PDFParser

T = TypeVar('T', bound=PDFParser)


class ParserRegistry:
    """Simplified registry for parsers."""
    
    _registry: Dict[str, Type[PDFParser]] = {}
    
    @classmethod
    def register_parser(cls, name: str, parser_class: Type[PDFParser]) -> None:
        """Register a parser class."""
        cls._registry[name] = parser_class

    @classmethod
    def parse(cls, name: str, pdf_path: Path, output_path: Path) -> str:
        """Parse PDF with specified parser."""
        parser_class = cls._registry.get(name)
        if not parser_class:
            raise ValueError(f"Unknown parser: {name}")
        
        parser = parser_class()
        return parser.parse(pdf_path, output_path)


def parser_registry() -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a parser class.
    
    Returns:
        Decorated parser class
    """
    def decorator(parser_class: Type[T]) -> Type[T]:
        # Get parser name using classmethod to avoid instantiation
        name = parser_class.parser_name()
        ParserRegistry.register_parser(name, parser_class)
        return parser_class
    return decorator