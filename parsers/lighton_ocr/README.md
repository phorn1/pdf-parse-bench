# LightOnOCR-2-1B Parser

PDF parser using [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) — a 1B-parameter end-to-end OCR model by LightOn based on Pixtral (vision encoder) and Qwen3 (LLM decoder).

## Setup

**Requirements:** NVIDIA GPU with CUDA support

**1. Install dependencies:**
```bash
uv pip install "transformers>=5.0.0" "torch>=2.0.1" "accelerate>=0.28.0"
```

**2. Run parser evaluation pipeline:**
```bash
uv run -m parsers.lighton_ocr
```
