# PaddleOCR-VL Parser

PDF parser using [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) .

## Setup

**Requirements:** NVIDIA GPU with CUDA support

**1. Install dependencies:**
```bash
uv pip install transformers torch pillow pymupdf
```

**2. Run parser:**
```bash
uv run -m parsers.paddle_ocr_vl
```
