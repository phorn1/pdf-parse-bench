# OpenAI Parsers

PDF parsers using OpenAI GPT models.

## Setup

**1. Install dependencies:**
```bash
uv pip install openai>=1.99.0
```

**2. Set API key in `.env`:**
```bash
OPENAI_API_KEY=your_api_key_here
```

**3. Run parser evaluation pipeline:**

For GPT-5:
```bash
uv run -m parsers.openai.gpt_5
```

For GPT-5 Mini:
```bash
uv run -m parsers.openai.gpt_5_mini
```

For GPT-5 Nano:
```bash
uv run -m parsers.openai.gpt_5_nano
```