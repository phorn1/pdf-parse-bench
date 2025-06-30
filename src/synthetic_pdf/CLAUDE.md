## Synthetic PDF Generator Module

This module generates HTML documents with mathematical formulas and text content using configurable styling, then converts them to PDFs using Playwright and Chromium. The primary use case is generating diverse document layouts for PDF parsing benchmarks.

The tool uses MathJax for LaTeX formula rendering and includes intelligent content fitting to ensure single-page PDF output.

## Module Structure

- `config.py` - Configuration management using Pydantic models
- `generators.py` - Text and formula content generators
- `html_builder.py` - HTML template generation and content block management
- `pdf_generator.py` - PDF generation using Playwright
- `validation.py` - Formula size validation utilities

## Configuration Files

- `../../config.yaml` - Main configuration file with styling presets
- `../../data/formulas.json` - Mathematical formulas database (LaTeX format)

## Output

- `../../artifacts/` - Output directory for generated HTML and PDF files

## Usage

This module is part of a uv project. Use `uv run` for executing Python scripts from the project root.