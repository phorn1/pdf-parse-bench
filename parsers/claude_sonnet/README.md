# Claude Sonnet Parser

PDF parser using [Claude Sonnet 4.6](https://docs.anthropic.com/en/docs/about-claude/models) via the Anthropic API.

## Setup

**1. Install dependencies:**
```bash
uv pip install anthropic
```

**2. Set API key in `.env`:**
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

**3. Run parser evaluation pipeline:**
```bash
uv run -m parsers.claude_sonnet
```