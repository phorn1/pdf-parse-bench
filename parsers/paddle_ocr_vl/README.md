# PaddleOCR-VL Parser

PDF parser using [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) (PaddlePaddle OCR Vision-Language).

## Setup

**Requirements:** NVIDIA GPU with CUDA 12.6 support

**1. Install PaddlePaddle (custom index):**
```bash
uv pip install paddlepaddle-gpu==3.2.0 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

**2. Install PaddleOCR:**
```bash
uv pip install "paddleocr[doc-parser]"
```

**3. Install safetensors (required development build):**
```bash
uv pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

**4. Run parser:**
```bash
uv run -m parsers.paddle_ocr_vl
```
