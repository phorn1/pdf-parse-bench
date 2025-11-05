# OlmOCR Parser

PDF parser using [OlmOCR](https://github.com/allenai/olmocr) via vLLM server.

## Setup

**Requirements:** NVIDIA GPU with ≥24GB VRAM

**1. Install dependencies:**
```bash
uv pip install vllm>=0.6.0 torch>=2.0.1
```

**2. Start vLLM server:**
```bash
uv run vllm serve allenai/olmOCR-2-7B-1025-FP8 \
  --served-model-name olmocr \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.80 \
  --trust-remote-code
```

**3. Run parser evaluation pipeline:**
```bash
uv run -m parsers.olmocr
```