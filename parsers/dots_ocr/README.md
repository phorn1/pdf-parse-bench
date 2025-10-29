# DOTS OCR Parser

PDF parser using [DOTS OCR](https://github.com/rednote-hilab/dots.ocr) (Rednote HiLab) via vLLM server.

## Setup

**Requirements:** NVIDIA GPU (1.7B parameter model - modest VRAM requirements)

**1. Start vLLM server:**
```bash
vllm serve rednote-hilab/dots.ocr \
  --trust-remote-code \
  --async-scheduling \
  --gpu-memory-utilization 0.95
```

**2. Run parser:**
```bash
uv run -m parsers.dots_ocr
```