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
VLLM_PLUGINS="" uv run -m parsers.deepseek_ocr -i data/<dataset>
```

> **Note:** `VLLM_PLUGINS=""` disables third-party vLLM plugins (e.g. PaddleX) that can cause CUDA initialization errors in the engine subprocess.
