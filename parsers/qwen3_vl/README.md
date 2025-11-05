# Qwen3-VL Parser

PDF parser using [Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) via OpenRouter.

## Setup

**1. Set API key in `.env`:**
```bash
OPENROUTER_API_KEY=your_api_key_here
```

**2. Run parser evaluation pipeline:**
```bash
uv run -m parsers.qwen3_vl
```