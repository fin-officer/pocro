# Use CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

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
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH="/app/src"

# Set timezone
ENV TZ=Europe/Berlin

# Set Python version
ENV PYTHON_VERSION=${PYTHON_VERSION}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python${PYTHON_VERSION}-venv \
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
    && rm -rf /var/lib/apt/lists/*

# Set Python version alternatives
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

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

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy application code and configuration
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY data/README.md ./data/README.md

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
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set the working directory to the app directory
WORKDIR /app/src

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]