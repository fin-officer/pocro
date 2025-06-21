"""
API routes for European Invoice OCR
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from api.dependencies import get_processor, get_settings, rate_limit
from config.settings import AppSettings as Settings
from core.pipeline import EuropeanInvoiceProcessor
from models.invoice_schema import InvoiceBatch, InvoiceData, InvoiceValidationResult
from models.validation import validate_invoice_data
from utils.monitoring import alert_manager, health_check, performance_tracker, system_monitor, track_processing_time

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()


# Response models
class ProcessingResponse(BaseModel):
    """Response model for invoice processing"""

    status: str
    filename: str
    extracted_data: Dict[str, Any]
    processing_time: float
    validation_result: Optional[InvoiceValidationResult] = None


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing"""

    status: str
    batch_id: str
    processed_count: int
    failed_count: int
    total_processing_time: float
    results: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Error response model"""

    status: str = "error"
    error: str
    details: Optional[Dict[str, Any]] = None


# Health and status endpoints
@router.get("/health", response_model=Dict[str, Any])
async def get_health():
    """
    Health check endpoint
    Returns system health status and basic metrics
    """
    try:
        health_status = health_check()
        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")


@router.get("/status", response_model=Dict[str, Any])
async def get_status(processor: EuropeanInvoiceProcessor = Depends(get_processor)):
    """
    Get detailed system status and model information
    """
    try:
        status = await processor.get_model_status()
        status.update(
            {
                "uptime_seconds": time.time() - system_monitor.start_time,
                "total_processed": performance_tracker.total_processed,
                "system_info": system_monitor.get_system_info(),
            }
        )

        return status

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")


@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(
    hours: int = Query(default=24, ge=1, le=168),  # 1 hour to 1 week
    processor: EuropeanInvoiceProcessor = Depends(get_processor),
):
    """
    Get processing metrics and performance statistics
    """
    try:
        metrics = await processor.get_metrics()

        # Add performance statistics
        perf_stats = performance_tracker.get_performance_stats(hours=hours)
        metrics["performance"] = perf_stats

        # Add system metrics
        metrics["system"] = system_monitor.get_system_info()

        return metrics

    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics retrieval failed")


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """
    Export metrics in Prometheus format
    """
    try:
        return system_monitor.export_prometheus_metrics()

    except Exception as e:
        logger.error(f"Prometheus metrics export failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics export failed")


@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_alerts():
    """
    Get current system alerts
    """
    try:
        alerts = alert_manager.check_alerts()
        return alerts

    except Exception as e:
        logger.error(f"Alert retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Alert retrieval failed")


# Processing endpoints
@router.post("/process-invoice", response_model=ProcessingResponse)
@track_processing_time("process_single_invoice")
async def process_single_invoice(
    file: UploadFile = File(...),
    validate: bool = Query(default=True, description="Validate extracted data"),
    processor: EuropeanInvoiceProcessor = Depends(get_processor),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit),
):
    """
    Process a single invoice file

    - **file**: Invoice file (PDF, JPG, JPEG, PNG)
    - **validate**: Whether to validate extracted data
    - **returns**: Extracted invoice data with processing metrics
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file type
    allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    file_ext = "." + file.filename.split(".")[-1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )

    # Check file size
    if hasattr(file, "size") and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=413, detail=f"File too large. Maximum size: {settings.max_file_size // (1024*1024)}MB"
        )

    try:
        # Process the invoice
        result = await processor.process_invoice_upload(file)

        # Validate extracted data if requested
        validation_result = None
        if validate and result.get("status") == "success":
            extracted_data = result.get("extracted_data", {})
            validation_result = validate_invoice_data(extracted_data)

        response = ProcessingResponse(
            status=result.get("status", "unknown"),
            filename=file.filename,
            extracted_data=result.get("extracted_data", {}),
            processing_time=result.get("processing_time", 0),
            validation_result=validation_result,
        )

        logger.info(f"Successfully processed {file.filename}")
        return response

    except Exception as e:
        logger.error(f"Failed to process {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/process-batch", response_model=BatchProcessingResponse)
@track_processing_time("process_batch_invoices")
async def process_batch_invoices(
    files: List[UploadFile] = File(...),
    validate: bool = Query(default=True, description="Validate extracted data"),
    max_concurrent: int = Query(default=3, ge=1, le=10, description="Maximum concurrent processes"),
    processor: EuropeanInvoiceProcessor = Depends(get_processor),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit),
):
    """
    Process multiple invoice files in batch

    - **files**: List of invoice files
    - **validate**: Whether to validate extracted data
    - **max_concurrent**: Maximum number of concurrent processes
    - **returns**: Batch processing results
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400, detail=f"Too many files. Maximum {settings.max_batch_size} files per batch"
        )

    start_time = time.time()
    batch_id = f"batch_{int(start_time)}"

    try:
        # Process files with limited concurrency
        semaphore = asyncio.Semaphore(min(max_concurrent, len(files)))

        async def process_single_file(file: UploadFile):
            async with semaphore:
                try:
                    result = await processor.process_invoice_upload(file)

                    # Add validation if requested
                    if validate and result.get("status") == "success":
                        extracted_data = result.get("extracted_data", {})
                        validation_result = validate_invoice_data(extracted_data)
                        result["validation_result"] = validation_result

                    return result

                except Exception as e:
                    logger.error(f"Failed to process {file.filename} in batch: {e}")
                    return {"status": "error", "filename": file.filename, "error": str(e), "processing_time": 0}

        # Execute batch processing
        tasks = [process_single_file(file) for file in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        failed_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_count += 1
                processed_results.append(
                    {"status": "error", "filename": files[i].filename, "error": str(result), "processing_time": 0}
                )
            else:
                if result.get("status") != "success":
                    failed_count += 1
                processed_results.append(result)

        total_processing_time = time.time() - start_time

        response = BatchProcessingResponse(
            status="completed",
            batch_id=batch_id,
            processed_count=len(files) - failed_count,
            failed_count=failed_count,
            total_processing_time=total_processing_time,
            results=processed_results,
        )

        logger.info(f"Batch {batch_id} completed: {len(files)} files, {failed_count} failed")
        return response

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.post("/validate", response_model=InvoiceValidationResult)
async def validate_invoice(invoice_data: Dict[str, Any], _: None = Depends(rate_limit)):
    """
    Validate invoice data against schema and business rules

    - **invoice_data**: Invoice data to validate
    - **returns**: Validation result with errors and warnings
    """
    try:
        validation_result = validate_invoice_data(invoice_data)
        return validation_result

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# Model management endpoints
@router.get("/models", response_model=Dict[str, Any])
async def get_model_info(processor: EuropeanInvoiceProcessor = Depends(get_processor)):
    """
    Get information about loaded models
    """
    try:
        model_status = await processor.get_model_status()
        return model_status

    except Exception as e:
        logger.error(f"Model info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Model info retrieval failed")


@router.post("/models/reload")
async def reload_models(
    background_tasks: BackgroundTasks, processor: EuropeanInvoiceProcessor = Depends(get_processor)
):
    """
    Reload models (background task)
    """
    try:

        async def reload_task():
            try:
                await processor.cleanup()
                await processor.initialize()
                logger.info("Models reloaded successfully")
            except Exception as e:
                logger.error(f"Model reload failed: {e}")

        background_tasks.add_task(reload_task)

        return {"status": "reload_initiated", "message": "Model reload started in background"}

    except Exception as e:
        logger.error(f"Model reload initiation failed: {e}")
        raise HTTPException(status_code=500, detail="Model reload failed")


# Configuration endpoints
@router.get("/config", response_model=Dict[str, Any])
async def get_configuration(settings: Settings = Depends(get_settings)):
    """
    Get current configuration (non-sensitive parts)
    """
    try:
        config = {
            "ocr_engine": settings.ocr_engine,
            "ocr_languages": settings.ocr_languages,
            "model_name": settings.model_name,
            "max_file_size_mb": settings.max_file_size // (1024 * 1024),
            "max_batch_size": settings.max_batch_size,
            "environment": settings.environment,
            "enable_metrics": settings.enable_metrics,
        }

        return config

    except Exception as e:
        logger.error(f"Configuration retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Configuration retrieval failed")


# Utility endpoints
@router.get("/languages", response_model=List[str])
async def get_supported_languages(settings: Settings = Depends(get_settings)):
    """
    Get list of supported OCR languages
    """
    return settings.ocr_languages


@router.get("/formats", response_model=Dict[str, List[str]])
async def get_supported_formats():
    """
    Get list of supported file formats
    """
    return {
        "input_formats": [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
        "output_formats": ["json", "xml", "csv"],
    }


@router.post("/test")
async def test_endpoint(test_data: Optional[Dict[str, Any]] = None, _: None = Depends(rate_limit)):
    """
    Test endpoint for API validation
    """
    return {"status": "ok", "message": "API is working correctly", "timestamp": time.time(), "received_data": test_data}


# Error handlers
@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail, details={"status_code": exc.status_code}).dict(),
    )


@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal server error", details={"type": type(exc).__name__}).dict(),
    )


# Include router with prefix
def create_router() -> APIRouter:
    """Create and configure the main API router"""
    main_router = APIRouter(prefix="/api/v1", tags=["invoice-ocr"])
    main_router.include_router(router)
    return main_router
