"""
Main processing pipeline for European invoice OCR
"""
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import os

import numpy as np
import cv2
from fastapi import UploadFile

from ..config.settings import Settings
from .preprocessor import InvoicePreprocessor
from .ocr_engine import MultilingualOCREngine
from .table_extractor import TableExtractor
from .llm_processor import LLMProcessor
from ..utils.file_utils import save_temp_file, cleanup_temp_file
from ..utils.monitoring import monitor_memory_usage

logger = logging.getLogger(__name__)


class EuropeanInvoiceProcessor:
    """Main processing pipeline for European invoices"""

    def __init__(self, settings: Settings):
        """Initialize the processing pipeline"""
        self.settings = settings

        # Initialize components
        self.preprocessor = InvoicePreprocessor()
        self.ocr_engine = None
        self.table_extractor = None
        self.llm_processor = None

        # Metrics
        self.total_processed = 0
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.total_processing_time = 0

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing European Invoice Processor...")

        try:
            # Initialize OCR engine
            self.ocr_engine = MultilingualOCREngine(
                languages=self.settings.ocr_languages,
                engine=self.settings.ocr_engine
            )
            logger.info("OCR engine initialized")

            # Initialize table extractor
            self.table_extractor = TableExtractor(use_ppstructure=True)
            logger.info("Table extractor initialized")

            # Initialize LLM processor
            self.llm_processor = LLMProcessor(
                model_name=self.settings.model_name,
                quantization=self.settings.quantization,
                use_vllm=True,
                max_model_len=self.settings.max_model_length
            )
            await self.llm_processor.initialize()
            logger.info("LLM processor initialized")

            logger.info("European Invoice Processor initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            raise

    async def process_invoice_upload(self, file: UploadFile) -> Dict[str, Any]:
        """Process an uploaded invoice file"""
        temp_file_path = None

        try:
            # Save uploaded file temporarily
            temp_file_path = await save_temp_file(file, self.settings.temp_dir)

            # Process the file
            result = await self.process_invoice_file(temp_file_path)

            return result

        except Exception as e:
            logger.error(f"Error processing uploaded file {file.filename}: {e}")
            raise
        finally:
            # Cleanup temp file
            if temp_file_path:
                cleanup_temp_file(temp_file_path)

    async def process_invoice_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single invoice file"""
        start_time = time.time()

        try:
            logger.info(f"Processing invoice file: {file_path}")

            # Step 1: Load and preprocess image
            image = await self._load_and_preprocess_image(file_path)
            logger.debug("Image preprocessing completed")

            # Step 2: Extract text with OCR
            ocr_results, detected_language = self.ocr_engine.extract_text(image, detect_language=True)
            full_text = self.ocr_engine.get_full_text(ocr_results)
            logger.debug(f"OCR completed, detected language: {detected_language}")

            # Step 3: Extract tables and line items
            tables = self.table_extractor.extract_tables(image)
            line_items = self.table_extractor.extract_line_items_from_ocr(ocr_results, image.shape[0])
            logger.debug(f"Table extraction completed, found {len(tables)} tables, {len(line_items)} line items")

            # Step 4: LLM-based structured extraction
            structured_data = await self.llm_processor.extract_structured_data(full_text, detected_language)
            logger.debug("LLM extraction completed")

            # Step 5: Enhance with table data
            enhanced_data = self._enhance_with_table_data(structured_data, line_items, tables)

            # Step 6: Add metadata
            processing_time = time.time() - start_time
            enhanced_data.update({
                "processing_metadata": {
                    "processing_time": processing_time,
                    "detected_language": detected_language,
                    "ocr_engine": self.settings.ocr_engine,
                    "model_name": self.settings.model_name,
                    "num_ocr_results": len(ocr_results),
                    "num_tables": len(tables),
                    "num_line_items": len(line_items),
                    "file_path": file_path
                }
            })

            # Update metrics
            self.total_processed += 1
            self.successful_extractions += 1
            self.total_processing_time += processing_time

            logger.info(f"Invoice processing completed in {processing_time:.2f}s")
            return enhanced_data

        except Exception as e:
            self.total_processed += 1
            self.failed_extractions += 1
            logger.error(f"Error processing invoice {file_path}: {e}")
            raise

    async def _load_and_preprocess_image(self, file_path: str) -> np.ndarray:
        """Load and preprocess image/PDF"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.pdf':
            # Convert PDF to images
            images = self.preprocessor.pdf_to_images(str(file_path))
            if not images:
                raise ValueError("No images extracted from PDF")
            # Use first page for now
            image = images[0]
        else:
            # Load image directly
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Could not load image from {file_path}")

        # Preprocess image
        processed_image = self.preprocessor.preprocess_invoice_image(image)

        return processed_image

    def _enhance_with_table_data(self, structured_data: Dict[str, Any],
                                line_items: List, tables: List) -> Dict[str, Any]:
        """Enhance structured data with table extraction results"""

        # Add line items if not present or improve existing ones
        if line_items:
            # Convert line items to dict format
            line_items_data = [item.to_dict() for item in line_items]

            if "invoice_lines" not in structured_data or not structured_data["invoice_lines"]:
                structured_data["invoice_lines"] = line_items_data
            else:
                # Try to merge/improve existing line items
                structured_data["invoice_lines"] = self._merge_line_items(
                    structured_data["invoice_lines"],
                    line_items_data
                )

        # Add table information
        if tables:
            structured_data["extracted_tables"] = [table.to_dict() for table in tables]

        return structured_data

    def _merge_line_items(self, llm_items: List[Dict], ocr_items: List[Dict]) -> List[Dict]:
        """Merge line items from LLM and OCR extraction"""
        # Simple merge strategy - use LLM items as base and fill missing data from OCR
        merged = llm_items.copy()

        # If LLM has fewer items than OCR, add missing ones
        if len(ocr_items) > len(merged):
            for i in range(len(merged), len(ocr_items)):
                merged.append(ocr_items[i])

        # Fill missing data in existing items
        for i, llm_item in enumerate(merged):
            if i < len(ocr_items):
                ocr_item = ocr_items[i]

                # Fill missing quantities
                if not llm_item.get("quantity") and ocr_item.get("quantity"):
                    llm_item["quantity"] = ocr_item["quantity"]

                # Fill missing prices
                if not llm_item.get("unit_price") and ocr_item.get("unit_price"):
                    llm_item["unit_price"] = ocr_item["unit_price"]

                if not llm_item.get("total_price") and ocr_item.get("total_price"):
                    llm_item["total_price"] = ocr_item["total_price"]

        return merged

    async def process_batch_upload(self, files: List[UploadFile]) -> List[Dict[str, Any]]:
        """Process multiple uploaded files"""
        results = []
        temp_files = []

        try:
            # Save all files temporarily
            for file in files:
                temp_path = await save_temp_file(file, self.settings.temp_dir)
                temp_files.append(temp_path)

            # Process files
            if len(temp_files) <= 5:  # Small batch - process concurrently
                tasks = [self.process_invoice_file(path) for path in temp_files]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:  # Large batch - process in chunks
                results = await self._process_large_batch(temp_files)

            # Convert exceptions to error dictionaries
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "error": str(result),
                        "file_index": i,
                        "status": "failed"
                    })
                else:
                    result["status"] = "success"
                    processed_results.append(result)

            return processed_results

        finally:
            # Cleanup all temp files
            for temp_path in temp_files:
                cleanup_temp_file(temp_path)

    async def _process_large_batch(self, file_paths: List[str], chunk_size: int = 5) -> List[Dict[str, Any]]:
        """Process large batch in chunks"""
        results = []

        for i in range(0, len(file_paths), chunk_size):
            chunk = file_paths[i:i + chunk_size]
            logger.info(f"Processing batch chunk {i//chunk_size + 1}/{(len(file_paths) + chunk_size - 1)//chunk_size}")

            # Process chunk
            tasks = [self.process_invoice_file(path) for path in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(chunk_results)

            # Small delay between chunks to prevent resource exhaustion
            await asyncio.sleep(0.1)

        return results

    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        memory_info = monitor_memory_usage()

        status = {
            "ocr_engine": {
                "engine": self.settings.ocr_engine,
                "languages": self.settings.ocr_languages,
                "status": "loaded" if self.ocr_engine else "not_loaded"
            },
            "llm_processor": {
                "model_name": self.settings.model_name,
                "quantization": self.settings.quantization,
                "status": "loaded" if self.llm_processor else "not_loaded"
            },
            "table_extractor": {
                "status": "loaded" if self.table_extractor else "not_loaded"
            },
            "memory": memory_info,
            "system_status": "ready" if all([self.ocr_engine, self.llm_processor, self.table_extractor]) else "initializing"
        }

        if self.llm_processor:
            status["llm_processor"].update(self.llm_processor.get_metrics())

        return status

    async def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        avg_processing_time = (self.total_processing_time / self.total_processed
                             if self.total_processed > 0 else 0)

        success_rate = (self.successful_extractions / self.total_processed
                       if self.total_processed > 0 else 0)

        metrics = {
            "processing_stats": {
                "total_processed": self.total_processed,
                "successful_extractions": self.successful_extractions,
                "failed_extractions": self.failed_extractions,
                "success_rate": round(success_rate * 100, 2),
                "avg_processing_time": round(avg_processing_time, 2),
                "total_processing_time": round(self.total_processing_time, 2)
            },
            "memory": monitor_memory_usage()
        }

        if self.llm_processor:
            metrics["llm_metrics"] = self.llm_processor.get_metrics()

        return metrics

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up processor resources...")

        if self.llm_processor:
            await self.llm_processor.cleanup()

        # Clear memory
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Processor cleanup completed")