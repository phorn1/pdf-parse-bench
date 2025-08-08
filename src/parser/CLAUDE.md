# Parser Module

This module provides a flexible, extensible framework for PDF parsing using various parser implementations.

## Architecture

### Core Components

- `core/base.py` - Abstract base classes and configuration
- `core/registry.py` - Parser registration system
- `core/__init__.py` - Core exports

### Parser Registry Pattern

All parsers use a decorator-based registration system:

```python
@parser_registry()  # For parsers using default ParserConfig
class MyParser(PDFParser):
    # Implementation

@parser_registry(MyParserConfig)  # For parsers with custom config
class MyAdvancedParser(PDFParser):
    # Implementation
```

### Auto-Discovery

The module automatically discovers and registers all parser implementations at import time using `pkgutil.iter_modules()`.

## Parser Categories

### Simple Parsers
- `pypdf_parser.py` - Basic PyPDF text extraction

### API-Based Parsers
- `adobe.py` - Adobe PDF Services (requires client_id, client_secret)
- `llamaparse.py` - LlamaParse API (requires api_key, supports premium_mode)
- `mistral.py` - Mistral OCR API (requires api_key, model)

### Advanced Parsers
- `marker_parser.py` - Multi-LLM service support with advanced configuration
- `vlm.py` - Vision Language Model with multi-provider support (OpenAI/Gemini)
- `unstructured_parser.py` - Unstructured library with configurable strategies
- `megaparse.py` - Advanced parsing with multiple strategies
