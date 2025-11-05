# Mathpix Parser

PDF parser using [Mathpix OCR](https://mathpix.com/).

## Setup

**1. Install dependencies:**
```bash
uv pip install mpxpy>=0.0.18
```

**2. Set API credentials in `.env`:**
```bash
MATHPIX_APP_ID=your_app_id_here
MATHPIX_APP_KEY=your_app_key_here
```

**3. Run parser evaluation pipeline:**
```bash
uv run -m parsers.mathpix
```