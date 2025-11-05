# Gemini Parsers

PDF parsers using Google Gemini models.

## Setup

**1. Install dependencies:**
```bash
uv pip install google-genai>=1.46.0
```

**2. Set API key in `.env`:**
```bash
GEMINI_API_KEY=your_api_key_here
```

**3. Run parser evaluation pipeline:**

For Gemini 2.5 Flash:
```bash
uv run -m parsers.gemini.gemini_2_5_flash
```

For Gemini 2.5 Pro:
```bash
uv run -m parsers.gemini.gemini_2_5_pro
```