# MonkeyOCR Parser

PDF parser using [MonkeyOCR](https://github.com/Ucas-HaoranWei/Monkey) via FastAPI.

## Setup

**Requirements:** Docker, NVIDIA GPU

**1. Start MonkeyOCR API server:**
```bash
docker compose up monkeyocr-api
```

**2. Run parser evaluation pipeline:**
```bash
uv run -m parsers.monkey_ocr
```
