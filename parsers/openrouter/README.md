# OpenRouter Parsers

PDF parsers using multimodal models via OpenRouter (OpenAI, Gemini, Anthropic, Qwen, Z.AI).

## Setup

**1. Set API key in `.env`:**
```bash
OPENROUTER_API_KEY=your_api_key_here
```

**2. Run parser evaluation pipeline:**

```bash
# OpenAI
uv run -m parsers.openrouter.gpt_5_2 -i data/<dataset>
uv run -m parsers.openrouter.gpt_5_mini -i data/<dataset>
uv run -m parsers.openrouter.gpt_5_nano -i data/<dataset>

# Google Gemini
uv run -m parsers.openrouter.gemini_2_5_flash -i data/<dataset>
uv run -m parsers.openrouter.gemini_2_5_pro -i data/<dataset>
uv run -m parsers.openrouter.gemini_3_flash -i data/<dataset>
uv run -m parsers.openrouter.gemini_3_pro -i data/<dataset>

# Anthropic Claude
uv run -m parsers.openrouter.sonnet_4_6 -i data/<dataset>
uv run -m parsers.openrouter.opus_4_6 -i data/<dataset>

# Vision-only models (use PDF→PNG conversion)
uv run -m parsers.openrouter.qwen3_vl -i data/<dataset>
uv run -m parsers.openrouter.glm_4_5v -i data/<dataset>
```
