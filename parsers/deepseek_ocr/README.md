# DeepSeek-OCR Parser

PDF parser using [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR).

## Setup

**Requirements:** NVIDIA GPU with ≥24GB VRAM

**1. Install dependencies:**
```bash
uv pip install vllm>=0.6.0 torch>=2.0.1
```

**2. Run parser evaluation pipeline:**
```bash
uv run -m parsers.deepseek_ocr
```