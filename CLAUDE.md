## Project Overview

This is a synthetic PDF generator tool for creating benchmark datasets. The system generates HTML documents with mathematical formulas and text content using configurable styling, then converts them to PDFs using Playwright and Chromium. The primary use case is generating diverse document layouts for PDF parsing benchmarks.

The tool uses MathJax for LaTeX formula rendering and includes intelligent content fitting to ensure single-page PDF output.

## Project Structure

- `src/config.py` - Configuration management using Pydantic models
- `src/generators.py` - Text and formula content generators
- `src/html_builder.py` - HTML template generation and content block management
- `src/pdf_generator.py` 
- `src/validation.py` - Formula size validation utilities
- `config.yaml` - Main configuration file with styling presets
- `data/formulas.json` - Mathematical formulas database (LaTeX format)
- `artifacts/` - Output directory for generated HTML and PDF files

## Usage

This is a uv project. Use `uv run` for executing Python scripts.

