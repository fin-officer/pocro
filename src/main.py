"""
European Invoice OCR API Main Application
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config.settings import AppSettings as Settings
from .core.pipeline import EuropeanInvoiceProcessor
from .models.invoice_schema import InvoiceData
from .utils.monitoring import monitor_memory_usage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global processor instance
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global processor

    # Startup
    logger.info("Starting European Invoice OCR API...")
    settings = Settings()
    # Debug: Print settings to verify they're loaded correctly
    logger.info(f"Application settings: {settings.dict()}")
    logger.info(f"MODEL_NAME from settings: {settings.model_name}")
    processor = EuropeanInvoiceProcessor(settings)
    await processor.initialize()
    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    if processor:
        await processor.cleanup()
    logger.info("Application shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="European Invoice OCR API",
    description="Local OCR processing for European invoices with LLM extraction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "European Invoice OCR API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        memory_info = monitor_memory_usage()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "memory": memory_info,
            "processor_ready": processor is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/process-invoice", response_model=dict)
async def process_invoice(file: UploadFile = File(...)):
    """Process a single invoice"""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    # Validate file type
    allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png"}
    file_ext = "." + file.filename.split(".")[-1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Allowed: {allowed_extensions}")

    try:
        # Process the invoice
        result = await processor.process_invoice_upload(file)

        return {
            "status": "success",
            "filename": file.filename,
            "extracted_data": result,
            "processing_time": result.get("processing_time", 0),
        }

    except Exception as e:
        logger.error(f"Error processing invoice {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/process-batch")
async def process_batch_invoices(files: List[UploadFile] = File(...)):
    """Process multiple invoices in batch"""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    max_files = 50
    if len(files) > max_files:
        raise HTTPException(status_code=400, detail=f"Too many files. Maximum {max_files} files per batch.")

    try:
        results = await processor.process_batch_upload(files)

        return {"status": "success", "processed_count": len(results), "results": results}

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.get("/models/status")
async def get_model_status():
    """Get status of loaded models"""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    return await processor.get_model_status()


@app.get("/metrics")
async def get_metrics():
    """Get processing metrics"""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    return await processor.get_metrics()


if __name__ == "__main__":
    import os

    port = int(os.getenv("PORT", "8005"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
