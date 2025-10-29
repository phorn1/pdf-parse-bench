# Use CUDA environment and install basic dependencies
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 git curl \
    wget poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Force create symbolic links (overwrite if exists)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Configure pip to use faster mirror and upgrade
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip setuptools || echo "Warning: pip upgrade failed"
ENV PIP_NO_CACHE_DIR=1

# Install PyTorch (version 2.6.0)
ARG CUDA_VERSION=126
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}

# Clone MonkeyOCR repository
RUN git clone https://github.com/Yuliang-Liu/MonkeyOCR.git /app/MonkeyOCR

# Set working directory to /app/MonkeyOCR
WORKDIR /app/MonkeyOCR

# Install project itself
RUN pip install -e .

# Install inference backend dependencies
RUN pip install lmdeploy==0.9.2

# Install FastAPI and uvicorn with standard extras for API service
RUN pip install "fastapi[standard]" "uvicorn[standard]"

# Install model download related dependencies
RUN pip install modelscope huggingface_hub

# Install PaddlePaddle and PaddleX for PP-DocLayout_plus-L support
RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu${CUDA_VERSION}/
RUN pip install "paddlex[base]"

# Copy local scripts (override repository versions with our custom ones)
COPY --chown=appuser:appuser download_models.sh /app/MonkeyOCR/
COPY --chown=appuser:appuser entrypoint.sh /app/MonkeyOCR/
RUN chmod +x /app/MonkeyOCR/download_models.sh /app/MonkeyOCR/entrypoint.sh

# Create model directory, temp directory, and set ownership
RUN mkdir -p /app/MonkeyOCR/model_weight /app/tmp && \
    chown -R appuser:appuser /app/MonkeyOCR /app/tmp

# Apply LMDeploy patcher
ARG LMDEPLOY_PATCHED=false
RUN if [ "$LMDEPLOY_PATCHED" = "true" ]; then \
      python /app/MonkeyOCR/tools/lmdeploy_patcher.py patch; \
      echo "Successfully apply lmdeploy patch" \
    else \
      echo "LMDEPLOY_PATCHED is false, skipping patch"; \
    fi

# Switch to non-root user
USER appuser

# Default command: start through entrypoint script
CMD ["/app/MonkeyOCR/entrypoint.sh"]
