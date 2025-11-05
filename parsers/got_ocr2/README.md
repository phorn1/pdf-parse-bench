# GOT-OCR2.0 Parser

PDF parser using [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0) (General OCR Theory).

## Setup

**Requirements:** NVIDIA GPU with CUDA support

**1. Install dependencies:**
```bash
uv pip install torch>=2.0.1 torchvision>=0.15.2 transformers==4.37.2 tiktoken>=0.6.0 verovio>=4.3.1 accelerate>=0.28.0
```

**2. Run parser evaluation pipeline:**
```bash
uv run -m parsers.got_ocr2
```