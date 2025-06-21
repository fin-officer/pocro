# Installation Guide

This guide will walk you through setting up the European Invoice OCR application.

## Prerequisites

- Python 3.10+
- pip (Python package manager)
- (Optional) CUDA-capable GPU for better performance
- (Optional) Docker and Docker Compose for containerized deployment

## Quick Start

1. **Clone the repository**
   ```bash
   git clone git@github.com:fin-officer/pocro.git
   cd pocro
   ```

2. **Set up environment**
   ```bash
   # Copy example environment file
   cp .env.example .env
   ```

   Edit the `.env` file to configure your settings. At minimum, set:
   ```
   MODEL_NAME=facebook/opt-125m  # or your preferred model
   ```

3. **Install dependencies**
   ```bash
   # Create and activate virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   The API will be available at `http://localhost:8000`

## Configuration

### Environment Variables

Key configuration options in `.env`:

- `MODEL_NAME`: The Hugging Face model to use (e.g., `facebook/opt-125m`)
- `QUANTIZATION`: Model quantization method (default: `awq`)
- `HOST`: Host to bind the server to (default: `0.0.0.0`)
- `PORT`: Port to run the server on (default: `8000`)
- `DEBUG`: Enable debug mode (default: `False`)
- `GPU_MEMORY_UTILIZATION`: GPU memory utilization (0-1) (default: `0.9`)

### Model Configuration

You can change the model by setting the `MODEL_NAME` environment variable in your `.env` file. The application supports any Hugging Face model compatible with the Transformers library.

Example:
```
MODEL_NAME=gpt2  # Use a different model
```

## Docker Installation

1. **Build the Docker image**
   ```bash
   docker build -t pocro .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 --env-file .env pocro
   ```

## Verifying the Installation

1. **Check the API**
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status":"ok"}`

2. **Test the API**
   ```bash
   curl -X POST http://localhost:8000/process \
     -H "Content-Type: application/json" \
     -d '{"text":"Sample invoice text"}'
   ```

## Troubleshooting

- **Model not loading**: Ensure `MODEL_NAME` is set to a valid Hugging Face model
- **CUDA errors**: Make sure you have the correct CUDA version installed if using GPU
- **Port in use**: Change the `PORT` in `.env` if the default port is occupied

For more help, see the [Troubleshooting Guide](./TROUBLESHOOTING.md).

## Next Steps

- [API Documentation](./API_DOCUMENTATION.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Troubleshooting](./TROUBLESHOOTING.md)