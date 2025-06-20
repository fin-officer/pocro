# Use full CUDA toolkit image with Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set build arguments
ARG PYTHON_VERSION=3.10

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.6.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    PATH="$POETRY_HOME/bin:$PATH"

# Set timezone
ENV TZ=Europe/Berlin

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-est \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-spa \
    tesseract-ocr-ita \
    tesseract-ocr-nld \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for Python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies using Poetry
RUN poetry install --no-interaction --no-ansi --no-root --only main

# Install PyTorch with CUDA 11.8 support
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available
RUN python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install additional CUDA libraries
RUN apt-get update && \
    apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    libcublas-11-8=11.11.3.6-1 \
    libcublas-dev-11-8=11.11.3.6-1 \
    libcufft-11-8=10.9.0.58-1 \
    libcufft-dev-11-8=10.9.0.58-1 \
    libcudnn8=8.9.2.*-1+cuda11.8 \
    libcudnn8-dev=8.9.2.*-1+cuda11.8 \
    && rm -rf /var/lib/apt/lists/*

# Set up environment variables for CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/cuda-11.8/bin:$PATH \
    CUDA_HOME=/usr/local/cuda-11.8

# Create symbolic links for CUDA libraries
RUN ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcufft.so.10 /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcufft.so.11 && \
    ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublas.so.11 /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublas.so.11.11.3.6 && \
    ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublasLt.so.11 /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublasLt.so.11.11.3.6 && \
    ldconfig

# Create a README.md file to prevent poetry install errors
RUN touch /app/README.md

# Copy application code and configuration
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml poetry.lock ./

# Ensure the start script is executable
RUN chmod +x /app/scripts/start.sh

# Install the package using Poetry
RUN poetry install --no-interaction --no-ansi --only main

# Create necessary directories with correct permissions
RUN mkdir -p /app/data/models /app/data/temp /app/data/output /app/logs && \
    chmod -R 777 /app/data /app/logs

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=4 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# Create non-root user and set permissions
RUN groupadd -r invoiceuser && \
    useradd -r -g invoiceuser -d /app -s /sbin/nologin invoiceuser && \
    chown -R invoiceuser:invoiceuser /app

# Switch to non-root user
USER invoiceuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Expose port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/scripts/start.sh"]

# Set the working directory to the app directory
WORKDIR /app/src

# Set the command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8005", "--reload", "--workers", "1"]