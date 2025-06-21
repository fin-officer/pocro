# Troubleshooting Guide

This guide provides solutions to common issues you might encounter while using the European Invoice OCR application.

## Environment and Setup Issues

### Model Not Loading

**Symptom**: The application fails to start with errors about model loading or "None is not a valid model".

**Solution**:
1. Check that `.env` exists and contains a valid `MODEL_NAME`:
   ```bash
   cat .env | grep MODEL_NAME
   ```
2. Ensure the model name is a valid Hugging Face model ID (e.g., `facebook/opt-125m`)
3. For private models, ensure you're authenticated:
   ```bash
   huggingface-cli login
   ```

### Environment Variables Not Loading

**Symptom**: Changes to `.env` are not taking effect.

**Solution**:
1. Verify the `.env` file is in the project root
2. Ensure variables are in UPPER_CASE
3. Restart the application after making changes
4. Check for typos in variable names

### CUDA/GPU Issues

**Symptom**: CUDA-related errors or warnings about falling back to CPU.

**Solution**:
1. Verify CUDA is installed:
   ```bash
   nvidia-smi  # Should show GPU info
   nvcc --version  # Should show CUDA version
   ```
2. Check PyTorch CUDA compatibility:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.version.cuda)  # Should match your CUDA version
   ```
3. Try setting `CUDA_VISIBLE_DEVICES` in `.env`:
   ```
   CUDA_VISIBLE_DEVICES=0
   ```

## API Issues

### Port Already in Use

**Symptom**: Error about port 8000 (or your configured port) being in use.

**Solution**:
1. Change the port in `.env`:
   ```
   PORT=8080
   ```
2. Or find and kill the process using the port:
   ```bash
   # Linux/macOS
   lsof -i :8000
   kill -9 <PID>
   
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

### CORS Errors

**Symptom**: Browser console shows CORS errors when making API requests.

**Solution**:
1. Ensure the frontend URL is in the `CORS_ORIGINS` list in `.env`:
   ```
   CORS_ORIGINS=http://localhost:3000,http://localhost:8080
   ```
2. For development, you can allow all origins (not recommended for production):
   ```
   CORS_ORIGINS=*
   ```

## Performance Issues

### Slow Processing

**Symptom**: The application is running slowly.

**Solution**:
1. Check GPU utilization:
   ```bash
   nvidia-smi -l 1  # Watch GPU usage
   ```
2. Try a smaller model or enable quantization in `.env`:
   ```
   QUANTIZATION=awq
   ```
3. Increase batch size if processing multiple documents:
   ```
   BATCH_SIZE=8
   ```

### High Memory Usage

**Symptom**: Application crashes with out-of-memory errors.

**Solution**:
1. Reduce GPU memory usage in `.env`:
   ```
   GPU_MEMORY_UTILIZATION=0.8
   ```
2. Use a smaller model
3. Enable gradient checkpointing if training:
   ```
   GRADIENT_CHECKPOINTING=True
   ```

## Debugging

### Enable Debug Logging

Set the log level in `.env`:
```
LOG_LEVEL=DEBUG
```

### Check Logs

Logs are written to `logs/app.log` by default. Check them for detailed error messages.

## Getting Help

If you've tried these solutions and are still experiencing issues:

1. Check the [GitHub Issues](https://github.com/finofficer/pocro/issues) for similar problems
2. Include the following in any bug reports:
   - Your `.env` file (with sensitive information removed)
   - The full error message and stack trace
   - Steps to reproduce the issue
   - Your environment (OS, Python version, CUDA version if using GPU)

## Common Error Messages

### "No module named 'transformers'"

Install the required packages:
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"

Reduce batch size or model size, or enable gradient accumulation.

### "ConnectionError: Couldn't reach server"

Check your internet connection and that Hugging Face Hub is accessible.