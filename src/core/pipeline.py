"""
Main processing pipeline for European Invoice OCR
"""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image

from src.config.settings import AppSettings as Settings
from src.core.llm_processor import LLMProcessor
from src.core.ocr_engine import InvoiceOCREngine
from src.core.preprocessor import InvoicePreprocessor
from src.core.table_extractor import InvoiceTableExtractor
from src.models.invoice_schema import InvoiceData, InvoiceItem
from src.models.validation import validate_invoice_data
from src.utils.file_utils import cleanup_temp_file, save_temp_file
from src.utils.monitoring import monitor_memory_usage, track_processing_time

logger = logging.getLogger(__name__)


class EuropeanInvoiceProcessor:
    """Main pipeline for processing European invoices"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.preprocessor = None
        self.ocr_engine = None
        self.table_extractor = None
        self.llm_processor = None

        # Processing statistics
        self.stats = {
            "processed_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_processing_time": 0,
            "avg_processing_time": 0,
        }

        logger.info("European Invoice Processor initialized")

    async def initialize(self):
        """Initialize all processing components"""
        try:
            logger.info("Initializing processing components...")

            # Initialize preprocessor
            self.preprocessor = InvoicePreprocessor()
            logger.info("✓ Preprocessor initialized")

            # Initialize OCR engine
            self.ocr_engine = InvoiceOCREngine(
                engine_type=self.settings.ocr_engine, languages=self.settings.ocr_languages
            )
            logger.info("✓ OCR engine initialized")

            # Initialize table extractor
            self.table_extractor = InvoiceTableExtractor(use_pp_structure=True)
            logger.info("✓ Table extractor initialized")

            # Initialize LLM processor
            self.llm_processor = LLMProcessor(
                model_name=self.settings.model_name, quantization=self.settings.quantization, use_vllm=True
            )
            await self.llm_processor.initialize()
            logger.info("✓ LLM processor initialized")

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def process_invoice_upload(self, file: UploadFile) -> Dict[str, Any]:
        """
        Process uploaded invoice file

        Args:
            file: Uploaded file

        Returns:
            Processing result
        """
        start_time = time.time()
        temp_file_path = None

        try:
            # Save uploaded file temporarily
            temp_file_path = await save_temp_file(file, self.settings.temp_dir)

            # Process the invoice
            result = await self.process_invoice_file(temp_file_path)

            # Add metadata
            result["filename"] = file.filename
            result["processing_time"] = time.time() - start_time
            result["file_size"] = file.size if hasattr(file, "size") else 0

            # Update statistics
            self._update_stats(True, result["processing_time"])

            logger.info(f"Successfully processed {file.filename} in {result['processing_time']:.2f}s")
            return result

        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "filename": file.filename,
                "processing_time": time.time() - start_time,
            }

            self._update_stats(False, time.time() - start_time)
            logger.error(f"Failed to process {file.filename}: {e}")
            return error_result

        finally:
            # Cleanup temporary file
            if temp_file_path:
                await cleanup_temp_file(temp_file_path)

    async def process_invoice_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process invoice file from disk

        Args:
            file_path: Path to invoice file

        Returns:
            Processing result
        """
        try:
            logger.info(f"Processing invoice file: {file_path}")

            # Step 1: Convert PDF to images if needed
            images = await self._load_invoice_images(file_path)

            # Step 2: Process each page/image
            all_results = []
            for i, image in enumerate(images):
                logger.debug(f"Processing page {i+1}/{len(images)}")

                page_result = await self._process_single_image(image, f"page_{i+1}")
                all_results.append(page_result)

            # Step 3: Combine results from all pages
            combined_result = self._combine_page_results(all_results)

            return {
                "status": "success",
                "pages_processed": len(images),
                "extracted_data": combined_result,
                "page_results": all_results,
            }

        except Exception as e:
            logger.error(f"Error processing invoice file {file_path}: {e}")
            raise

    async def _load_invoice_images(self, file_path: str) -> List[np.ndarray]:
        """Load and convert invoice file to images"""
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".pdf":
                # Convert PDF to images
                pil_images = self.preprocessor.pdf_to_images(file_path)
                images = [np.array(img) for img in pil_images]
                logger.debug(f"Converted PDF to {len(images)} images")

            elif file_ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
                # Load single image
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"Could not load image from {file_path}")
                images = [image]
                logger.debug("Loaded single image")

            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            return images

        except Exception as e:
            logger.error(f"Failed to load images from {file_path}: {e}")
            raise

    async def _process_single_image(self, image: np.ndarray, page_id: str) -> Dict[str, Any]:
        """Process a single invoice image"""
        try:
            # Step 1: Preprocess image
            logger.debug(f"Preprocessing {page_id}")
            preprocessed_image = self.preprocessor.preprocess_invoice_image(image)

            # Step 2: OCR text extraction
            logger.debug(f"Extracting text from {page_id}")
            ocr_result = self.ocr_engine.extract_invoice_text(preprocessed_image)

            # Step 3: Table extraction
            logger.debug(f"Extracting tables from {page_id}")
            tables = self.table_extractor.extract_tables(preprocessed_image, ocr_result["text_elements"])

            # Step 4: Parse line items from tables
            line_items = self.table_extractor.parse_invoice_line_items(tables)

            # Step 5: Detect language
            detected_language = self.ocr_engine.detect_language(ocr_result["text_elements"])

            # Step 6: LLM structured extraction
            logger.debug(f"Extracting structured data from {page_id}")
            structured_data = await self.llm_processor.extract_structured_data(
                ocr_result["full_text"], detected_language
            )

            # Step 7: Merge table data with LLM data
            merged_data = self._merge_extraction_results(structured_data, line_items, tables)

            return {
                "page_id": page_id,
                "ocr_result": ocr_result,
                "tables": tables,
                "line_items": line_items,
                "detected_language": detected_language,
                "structured_data": merged_data,
                "quality_metrics": self._calculate_quality_metrics(ocr_result, structured_data),
            }

        except Exception as e:
            logger.error(f"Error processing {page_id}: {e}")
            return {
                "page_id": page_id,
                "error": str(e),
                "ocr_result": {"full_text": "", "text_elements": []},
                "structured_data": {},
            }

    def _merge_extraction_results(self, llm_data: Dict, line_items: List[Dict], tables: List[Dict]) -> Dict:
        """Merge LLM extraction with table extraction results"""
        try:
            # Start with LLM data as base
            merged = llm_data.copy()

            # Enhance line items with table data if available
            if line_items and not merged.get("invoice_lines"):
                merged["invoice_lines"] = line_items
            elif line_items and merged.get("invoice_lines"):
                # Compare and potentially merge line items
                merged["invoice_lines"] = self._merge_line_items(merged["invoice_lines"], line_items)

            # Add table metadata
            merged["tables_detected"] = len(tables)
            merged["table_line_items"] = len(line_items)

            # Validate financial totals
            merged = self._validate_financial_totals(merged)

            return merged

        except Exception as e:
            logger.error(f"Error merging extraction results: {e}")
            return llm_data

    def _merge_line_items(self, llm_items: List[Dict], table_items: List[Dict]) -> List[Dict]:
        """Merge line items from LLM and table extraction"""
        # For now, prefer table extraction for line items as it's more structured
        if table_items:
            return table_items
        return llm_items

    def _validate_financial_totals(self, data: Dict) -> Dict:
        """Validate and correct financial totals"""
        try:
            line_items = data.get("invoice_lines", [])

            if line_items:
                # Calculate totals from line items
                calculated_subtotal = sum(float(item.get("line_total", 0)) for item in line_items)

                # Update totals if they seem incorrect
                if not data.get("total_excl_vat") or abs(data["total_excl_vat"] - calculated_subtotal) > 0.01:
                    data["total_excl_vat"] = calculated_subtotal

                    # Estimate VAT if not present
                    if not data.get("total_vat"):
                        # Assume 20% VAT as default for European invoices
                        data["total_vat"] = calculated_subtotal * 0.20

                    # Calculate total including VAT
                    data["total_incl_vat"] = data["total_excl_vat"] + data["total_vat"]

            return data

        except Exception as e:
            logger.error(f"Error validating financial totals: {e}")
            return data

    def _calculate_quality_metrics(self, ocr_result: Dict, structured_data: Dict) -> Dict:
        """Calculate quality metrics for the extraction"""
        metrics = {
            "ocr_confidence": ocr_result.get("avg_confidence", 0),
            "text_elements_count": ocr_result.get("total_elements", 0),
            "required_fields_present": 0,
            "data_completeness": 0,
            "overall_quality": 0,
        }

        # Check required fields
        required_fields = ["invoice_id", "issue_date", "supplier", "total_incl_vat"]
        present_fields = sum(1 for field in required_fields if structured_data.get(field))
        metrics["required_fields_present"] = present_fields / len(required_fields)

        # Calculate data completeness
        all_fields = ["invoice_id", "issue_date", "supplier", "customer", "invoice_lines", "total_incl_vat"]
        complete_fields = sum(1 for field in all_fields if structured_data.get(field))
        metrics["data_completeness"] = complete_fields / len(all_fields)

        # Overall quality score
        metrics["overall_quality"] = (
            metrics["ocr_confidence"] * 0.3
            + metrics["required_fields_present"] * 0.4
            + metrics["data_completeness"] * 0.3
        )

        return metrics

    def _combine_page_results(self, page_results: List[Dict]) -> Dict:
        """Combine results from multiple pages"""
        if not page_results:
            return {}

        if len(page_results) == 1:
            return page_results[0]["structured_data"]

        # For multi-page invoices, combine data intelligently
        combined = page_results[0]["structured_data"].copy()

        # Combine line items from all pages
        all_line_items = []
        for result in page_results:
            line_items = result.get("structured_data", {}).get("invoice_lines", [])
            all_line_items.extend(line_items)

        combined["invoice_lines"] = all_line_items

        # Recalculate totals
        combined = self._validate_financial_totals(combined)

        return combined

    async def process_batch_upload(self, files: List[UploadFile]) -> List[Dict]:
        """Process multiple invoice files"""
        results = []

        logger.info(f"Processing batch of {len(files)} files")

        # Process files in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent processes

        async def process_single_file(file):
            async with semaphore:
                return await self.process_invoice_upload(file)

        # Create tasks for all files
        tasks = [process_single_file(file) for file in files]

        # Execute with progress tracking
        completed = 0
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed += 1
            logger.info(f"Batch progress: {completed}/{len(files)} files processed")

        logger.info(f"Batch processing completed: {len(results)} files processed")
        return results

    def _update_stats(self, success: bool, processing_time: float):
        """Update processing statistics"""
        self.stats["processed_count"] += 1
        self.stats["total_processing_time"] += processing_time

        if success:
            self.stats["success_count"] += 1
        else:
            self.stats["error_count"] += 1

        self.stats["avg_processing_time"] = self.stats["total_processing_time"] / self.stats["processed_count"]

    async def get_model_status(self) -> Dict:
        """Get status of loaded models"""
        status = {
            "ocr_engine": {
                "type": self.settings.ocr_engine,
                "languages": self.settings.ocr_languages,
                "initialized": self.ocr_engine is not None,
            },
            "llm_model": {
                "name": self.settings.model_name,
                "quantization": self.settings.quantization,
                "initialized": self.llm_processor is not None,
            },
            "memory_usage": monitor_memory_usage(),
        }

        return status

    async def get_metrics(self) -> Dict:
        """Get processing metrics"""
        return {
            "processing_stats": self.stats,
            "system_metrics": monitor_memory_usage(),
            "settings": {
                "ocr_engine": self.settings.ocr_engine,
                "model_name": self.settings.model_name,
                "max_file_size": self.settings.max_file_size,
                "max_batch_size": self.settings.max_batch_size,
            },
        }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up processor resources...")

        if self.llm_processor:
            await self.llm_processor.cleanup()

        # Clear any temporary files
        temp_dir = Path(self.settings.temp_dir)
        if temp_dir.exists():
            for temp_file in temp_dir.glob("temp_*"):
                try:
                    temp_file.unlink()
                except Exception:
                    pass

        logger.info("Processor cleanup completed")
