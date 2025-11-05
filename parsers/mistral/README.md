# Mistral OCR Parser

PDF parser using [Mistral OCR](https://mistral.ai/).

## Setup

**1. Install dependencies:**
```bash
uv pip install mistralai>=1.9.0
```

**2. Set API key in `.env`:**
```bash
MISTRAL_API_KEY=your_api_key_here
```

**3. Run parser evaluation pipeline:**
```bash
uv run -m parsers.mistral
```