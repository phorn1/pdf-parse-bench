# LlamaParse Parser

PDF parser using [LlamaParse](https://github.com/run-llama/llama_parse).

## Setup

**1. Install dependencies:**
```bash
uv pip install llama-cloud-services>=0.6.76
```

**2. Set API key in `.env`:**
```bash
LLAMACLOUD_API_KEY=your_api_key_here
```

**3. Run parser evaluation pipeline:**
```bash
uv run -m parsers.llamaparse
```