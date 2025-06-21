"""
LLM processor for structured invoice data extraction
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from ..models.invoice_schema import InvoiceData
from ..prompts.extraction_prompts import get_extraction_prompt

logger = logging.getLogger(__name__)


class LLMProcessor:
    """LLM processor for invoice data extraction"""

    def __init__(self, model_name: str, quantization: str = "awq", use_vllm: bool = True, max_model_len: int = 4096):
        """
        Initialize LLM processor

        Args:
            model_name: Name of the model to use
            quantization: Quantization method (awq, nf4, gptq)
            use_vllm: Whether to use vLLM for inference
            max_model_len: Maximum model context length
        """
        self.model_name = model_name
        self.quantization = quantization
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.max_model_len = max_model_len

        self.model = None
        self.tokenizer = None
        self.sampling_params = None

        # Model name mappings
        # Use the provided model_name directly, fall back to mappings if needed
        self.model_mappings = {
            # Keep existing mappings for backward compatibility
            "mistral-7b-instruct": model_name,  # Use the provided model_name
            "qwen2.5-7b": model_name,
            "llama-3.1-8b": model_name,
        }

        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_time = 0

    async def initialize(self):
        """Initialize the LLM model"""
        logger.info(f"Initializing LLM processor with {self.model_name}")

        if self.use_vllm:
            await self._initialize_vllm()
        else:
            await self._initialize_transformers()

        logger.info("LLM processor initialized successfully")

    async def _initialize_vllm(self):
        """Initialize vLLM model"""
        try:
            # Use the model_name directly instead of the mapping
            model_id = self.model_name

            # Configure vLLM
            self.model = LLM(
                model=model_id,
                quantization=self.quantization if self.quantization != "nf4" else None,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=0.9,
                dtype="half",
                trust_remote_code=True,
            )

            # Configure sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.1, top_p=0.9, max_tokens=1024, stop_token_ids=[2], repetition_penalty=1.1  # EOS token
            )

            logger.info("vLLM model initialized")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            logger.info("Falling back to transformers...")
            self.use_vllm = False
            await self._initialize_transformers()

    async def _initialize_transformers(self):
        """Initialize transformers model"""
        try:
            model_id = self.model_mappings.get(self.model_name, self.model_name)

            # Configure quantization
            quantization_config = None
            if self.quantization == "nf4":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            logger.info("Transformers model initialized")

        except Exception as e:
            logger.error(f"Failed to initialize transformers model: {e}")
            raise

    async def extract_structured_data(self, invoice_text: str, detected_language: str = "en") -> Dict[str, Any]:
        """
        Extract structured data from invoice text

        Args:
            invoice_text: Raw text from OCR
            detected_language: Detected language of the invoice

        Returns:
            Dictionary containing extracted invoice data
        """
        start_time = time.time()

        try:
            # Create prompt
            prompt = create_multilingual_prompt(invoice_text, detected_language)

            # Generate response
            if self.use_vllm:
                response = await self._generate_vllm(prompt)
            else:
                response = await self._generate_transformers(prompt)

            # Parse JSON response
            structured_data = self._parse_json_response(response)

            # Validate with Pydantic model
            try:
                validated_data = InvoiceData(**structured_data)
                result = validated_data.dict()
            except Exception as e:
                logger.warning(f"Validation failed: {e}, using raw extraction")
                result = structured_data

            # Update metrics
            processing_time = time.time() - start_time
            self.total_requests += 1
            self.total_time += processing_time

            result["processing_time"] = processing_time
            result["confidence_score"] = self._calculate_confidence_score(result)

            return result

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._fallback_extraction(invoice_text)

    async def _generate_vllm(self, prompt: str) -> str:
        """Generate response using vLLM"""
        try:
            outputs = self.model.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise

    async def _generate_transformers(self, prompt: str) -> str:
        """Generate response using transformers"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_model_len - 1024,  # Leave space for output
                padding=True,
            )

            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            logger.error(f"Transformers generation failed: {e}")
            raise

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            # Try to find JSON in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # If no JSON found, try parsing entire response
                return json.loads(response)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response}")
            raise

    def _calculate_confidence_score(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted data"""
        # Simple confidence calculation based on completeness
        required_fields = ["invoice_id", "issue_date", "supplier", "total_incl_vat"]
        present_fields = sum(1 for field in required_fields if data.get(field))

        base_confidence = present_fields / len(required_fields)

        # Bonus for having line items
        if data.get("invoice_lines") and len(data["invoice_lines"]) > 0:
            base_confidence = min(1.0, base_confidence + 0.1)

        return round(base_confidence, 2)

    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback extraction using simple patterns"""
        import re

        # Simple pattern-based extraction
        result = {
            "invoice_id": None,
            "issue_date": None,
            "supplier": {"name": None},
            "total_incl_vat": None,
            "raw_text": text,
            "extraction_method": "fallback",
            "confidence_score": 0.3,
        }

        # Extract invoice number
        invoice_patterns = [
            r"invoice\s*(?:no\.?|number)?\s*:?\s*([A-Z0-9-]+)",
            r"rechnung\s*(?:nr\.?)?\s*:?\s*([A-Z0-9-]+)",
            r"arve\s*(?:nr\.?)?\s*:?\s*([A-Z0-9-]+)",
        ]

        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["invoice_id"] = match.group(1)
                break

        # Extract date
        date_pattern = r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
        date_match = re.search(date_pattern, text)
        if date_match:
            result["issue_date"] = date_match.group()

        # Extract total amount
        amount_patterns = [
            r"total\s*:?\s*[\€\$\£]?\s*(\d+[.,]\d{2})",
            r"gesamt\s*:?\s*[\€\$\£]?\s*(\d+[.,]\d{2})",
            r"kokku\s*:?\s*[\€\$\£]?\s*(\d+[.,]\d{2})",
        ]

        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(",", ".")
                try:
                    result["total_incl_vat"] = float(amount_str)
                    break
                except ValueError:
                    continue

        return result

    async def process_batch(self, invoice_texts: List[str], detected_languages: List[str]) -> List[Dict[str, Any]]:
        """Process multiple invoices in batch"""
        if self.use_vllm:
            return await self._process_batch_vllm(invoice_texts, detected_languages)
        else:
            # For transformers, process sequentially to avoid memory issues
            results = []
            for text, lang in zip(invoice_texts, detected_languages):
                result = await self.extract_structured_data(text, lang)
                results.append(result)
            return results

    async def _process_batch_vllm(
        self, invoice_texts: List[str], detected_languages: List[str]
    ) -> List[Dict[str, Any]]:
        """Process batch using vLLM"""
        try:
            # Create prompts
            prompts = [create_multilingual_prompt(text, lang) for text, lang in zip(invoice_texts, detected_languages)]

            # Generate responses
            outputs = self.model.generate(prompts, self.sampling_params)

            # Parse responses
            results = []
            for i, output in enumerate(outputs):
                try:
                    response = output.outputs[0].text.strip()
                    structured_data = self._parse_json_response(response)

                    # Validate
                    try:
                        validated_data = InvoiceData(**structured_data)
                        result = validated_data.dict()
                    except Exception:
                        result = structured_data

                    result["confidence_score"] = self._calculate_confidence_score(result)
                    results.append(result)

                except Exception as e:
                    logger.error(f"Failed to process batch item {i}: {e}")
                    results.append(self._fallback_extraction(invoice_texts[i]))

            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fall back to sequential processing
            results = []
            for text, lang in zip(invoice_texts, detected_languages):
                result = await self.extract_structured_data(text, lang)
                results.append(result)
            return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        avg_time = self.total_time / self.total_requests if self.total_requests > 0 else 0

        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "avg_time_per_request": avg_time,
            "model_name": self.model_name,
            "quantization": self.quantization,
            "backend": "vllm" if self.use_vllm else "transformers",
        }

    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self.model, "cleanup"):
            self.model.cleanup()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
