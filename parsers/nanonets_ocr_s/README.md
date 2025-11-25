# Nanonets OCR-s Parser

PDF parser using [Nanonets OCR-s](https://huggingface.co/nanonets/Nanonets-OCR-s).

## Setup

**Requirements:** NVIDIA GPU with CUDA support

**1. Install dependencies:**
```bash
uv pip install torch>=2.0.1 transformers>=4.49.0 accelerate>=0.28.0
```

**2. Run parser evaluation pipeline:**
```bash
uv run -m parsers.nanonets_ocr_s
```