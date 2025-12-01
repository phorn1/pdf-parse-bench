# GROBID Parser

PDF parser using [GROBID](https://github.com/kermitt2/grobid).

## Setup

**Requirements:** NVIDIA GPU (for Deep Learning models)

**1. Start GROBID server:**
```bash
docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2-full
```

**2. Install dependencies:**
```bash
uv pip install requests grobid-tei-xml
```

**3. Run parser evaluation pipeline:**
```bash
uv run -m parsers.grobid
```
