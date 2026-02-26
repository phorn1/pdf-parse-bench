# Chandra Parser

PDF parser using [Chandra OCR](https://huggingface.co/datalab-to/chandra) — a 9B parameter model supporting 40+ languages, handwriting, forms, tables, and complex layouts.

## Setup

**Requirements:** NVIDIA GPU with CUDA support

**1. Install dependencies:**
```bash
uv pip install chandra-ocr
```

**2. Run parser evaluation pipeline:**
```bash
uv run -m parsers.chandra
```
