# LlamaParse Parser

PDF parser using [LlamaParse](https://docs.cloud.llamaindex.ai).

## Setup

**1. Install dependencies:**
```bash
uv pip install llama-cloud>=1.0
```

**2. Set API key in `.env`:**
```bash
LLAMACLOUD_API_KEY=your_api_key_here
```

**3. Run parser evaluation pipeline:**
```bash
uv run -m parsers.llamaparse
```