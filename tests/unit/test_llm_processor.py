"""
Unit tests for LLM processor
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.core.llm_processor import LLMProcessor
from src.models.invoice_schema import InvoiceData


class TestLLMProcessor:
    """Test cases for LLMProcessor class"""
    
    @pytest.fixture
    def sample_invoice_text(self):
        """Sample invoice text for testing"""
        return """
        INVOICE #INV-2024-001
        Date: 15.01.2024
        
        From:
        Test Company GmbH
        VAT ID: DE123456789
        
        To:
        Customer Ltd
        VAT ID: EE987654321
        
        Description     Qty    Price    Total
        Software        1      100.00   100.00
        Support         1       50.00    50.00
        
        Subtotal:               150.00
        VAT (19%):               28.50
        Total:                  178.50
        """
    
    @pytest.fixture
    def sample_llm_response(self):
        """Sample LLM JSON response"""
        return '''
        {
            "invoice_id": "INV-2024-001",
            "issue_date": "2024-01-15",
            "currency_code": "EUR",
            "supplier": {
                "name": "Test Company GmbH",
                "vat_id": "DE123456789",
                "country_code": "DE"
            },
            "customer": {
                "name": "Customer Ltd",
                "vat_id": "EE987654321",
                "country_code": "EE"
            },
            "invoice_lines": [
                {
                    "line_id": "1",
                    "description": "Software",
                    "quantity": 1.0,
                    "unit_price": 100.00,
                    "line_total": 100.00,
                    "vat_rate": 0.19
                },
                {
                    "line_id": "2",
                    "description": "Support",
                    "quantity": 1.0,
                    "unit_price": 50.00,
                    "line_total": 50.00,
                    "vat_rate": 0.19
                }
            ],
            "total_excl_vat": 150.00,
            "total_vat": 28.50,
            "total_incl_vat": 178.50
        }
        '''
    
    @pytest.fixture
    def invalid_llm_response(self):
        """Invalid LLM response for testing error handling"""
        return '''
        {
            "invoice_id": "INV-2024-001",
            "issue_date": "invalid-date",
            "currency_code": "INVALID",
            "supplier": {
                "name": "",
                "vat_id": "INVALID123"
            },
            "total_excl_vat": -100,
            "total_vat": "not_a_number",
            "total_incl_vat": 50
        }
        '''
    
    def test_llm_processor_initialization_vllm(self):
        """Test LLM processor initialization with vLLM"""
        with patch('src.core.llm_processor.VLLM_AVAILABLE', True):
            processor = LLMProcessor(
                model_name="mistral-7b-instruct",
                quantization="awq",
                use_vllm=True
            )
            assert processor.model_name == "mistral-7b-instruct"
            assert processor.quantization == "awq"
            assert processor.use_vllm is True
    
    def test_llm_processor_initialization_transformers(self):
        """Test LLM processor initialization with Transformers"""
        with patch('src.core.llm_processor.TRANSFORMERS_AVAILABLE', True):
            processor = LLMProcessor(
                model_name="mistral-7b-instruct",
                quantization="nf4",
                use_vllm=False
            )
            assert processor.model_name == "mistral-7b-instruct"
            assert processor.quantization == "nf4"
            assert processor.use_vllm is False
    
    @pytest.mark.asyncio
    async def test_initialize_vllm(self):
        """Test vLLM initialization"""
        with patch('src.core.llm_processor.VLLM_AVAILABLE', True):
            with patch('src.core.llm_processor.LLM') as mock_llm:
                with patch('src.core.llm_processor.SamplingParams') as mock_params:
                    mock_llm.return_value = Mock()
                    mock_params.return_value = Mock()
                    
                    processor = LLMProcessor("mistral-7b-instruct", use_vllm=True)
                    await processor.initialize()
                    
                    mock_llm.assert_called_once()
                    mock_params.assert_called_once()
                    assert processor.model is not None
                    assert processor.sampling_params is not None
    
    @pytest.mark.asyncio
    async def test_initialize_transformers(self):
        """Test Transformers initialization"""
        with patch('src.core.llm_processor.TRANSFORMERS_AVAILABLE', True):
            with patch('src.core.llm_processor.AutoModelForCausalLM') as mock_model:
                with patch('src.core.llm_processor.AutoTokenizer') as mock_tokenizer:
                    with patch('src.core.llm_processor.BitsAndBytesConfig') as mock_config:
                        mock_model.from_pretrained.return_value = Mock()
                        mock_tokenizer.from_pretrained.return_value = Mock()
                        mock_config.return_value = Mock()
                        
                        processor = LLMProcessor("mistral-7b-instruct", quantization="nf4", use_vllm=False)
                        await processor.initialize()
                        
                        mock_model.from_pretrained.assert_called_once()
                        mock_tokenizer.from_pretrained.assert_called_once()
                        assert processor.model is not None
                        assert processor.tokenizer is not None
    
    @pytest.mark.asyncio
    async def test_extract_structured_data_success(self, sample_invoice_text, sample_llm_response):
        """Test successful structured data extraction"""
        with patch('src.core.llm_processor.VLLM_AVAILABLE', True):
            processor = LLMProcessor("mistral-7b-instruct", use_vllm=True)
            processor.model = Mock()
            processor.sampling_params = Mock()
            
            # Mock vLLM generation
            mock_output = Mock()
            mock_output.outputs = [Mock()]
            mock_output.outputs[0].text = sample_llm_response
            processor.model.generate.return_value = [mock_output]
            
            result = await processor.extract_structured_data(sample_invoice_text, "en")
            
            assert isinstance(result, dict)
            assert result['invoice_id'] == "INV-2024-001"
            assert result['issue_date'] == "2024-01-15"
            assert result['currency_code'] == "EUR"
            assert len(result['invoice_lines']) == 2
            assert result['total_incl_vat'] == 178.50
    
    @pytest.mark.asyncio
    async def test_extract_structured_data_with_transformers(self, sample_invoice_text, sample_llm_response):
        """Test structured data extraction with Transformers"""
        with patch('src.core.llm_processor.TRANSFORMERS_AVAILABLE', True):
            with patch('src.core.llm_processor.torch') as mock_torch:
                processor = LLMProcessor("mistral-7b-instruct", use_vllm=False)
                processor.model = Mock()
                processor.tokenizer = Mock()
                
                # Mock tokenizer
                processor.tokenizer.return_value = {
                    'input_ids': mock_torch.tensor([[1, 2, 3]]),
                    'attention_mask': mock_torch.tensor([[1, 1, 1]])
                }
                processor.tokenizer.pad_token_id = 0
                processor.tokenizer.eos_token_id = 2
                
                # Mock model generation
                mock_torch.tensor.return_value.to.return_value = mock_torch.tensor([[1, 2, 3]])
                processor.model.device = 'cpu'
                processor.model.generate.return_value = mock_torch.tensor([[1, 2, 3, 4, 5]])
                processor.tokenizer.decode.return_value = sample_llm_response
                
                result = await processor.extract_structured_data(sample_invoice_text, "en")
                
                assert isinstance(result, dict)
                assert result['invoice_id'] == "INV-2024-001"
    
    @pytest.mark.asyncio
    async def test_extract_structured_data_json_error(self, sample_invoice_text):
        """Test structured data extraction with JSON parsing error"""
        with patch('src.core.llm_processor.VLLM_AVAILABLE', True):
            processor = LLMProcessor("mistral-7b-instruct", use_vllm=True)
            processor.model = Mock()
            processor.sampling_params = Mock()
            
            # Mock invalid JSON response
            mock_output = Mock()
            mock_output.outputs = [Mock()]
            mock_output.outputs[0].text = "invalid json response"
            processor.model.generate.return_value = [mock_output]
            
            result = await processor.extract_structured_data(sample_invoice_text, "en")
            
            # Should return fallback response
            assert isinstance(result, dict)
            assert 'raw_text' in result
    
    def test_parse_llm_response_valid_json(self, sample_llm_response):
        """Test parsing valid JSON response"""
        processor = LLMProcessor("test-model")
        
        result = processor._parse_llm_response(sample_llm_response)
        
        assert isinstance(result, dict)
        assert result['invoice_id'] == "INV-2024-001"
        assert result['total_incl_vat'] == 178.50
    
    def test_parse_llm_response_json_in_text(self):
        """Test parsing JSON embedded in text"""
        processor = LLMProcessor("test-model")
        
        response_with_text = '''
        Here is the extracted invoice data:
        
        {
            "invoice_id": "INV-2024-001",
            "issue_date": "2024-01-15",
            "currency_code": "EUR"
        }
        
        This completes the extraction.
        '''
        
        result = processor._parse_llm_response(response_with_text)
        
        assert isinstance(result, dict)
        assert result['invoice_id'] == "INV-2024-001"
    
    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON response"""
        processor = LLMProcessor("test-model")
        
        invalid_response = "This is not JSON at all"
        
        result = processor._parse_llm_response(invalid_response)
        
        assert result == {}  # Should return empty dict on error
    
    def test_clean_json_response(self):
        """Test JSON response cleaning"""
        processor = LLMProcessor("test-model")
        
        # Test markdown removal
        dirty_json = '''```json
        {
            "invoice_id": "INV-001",
            "amount": 100.50,
        }
        ```'''
        
        cleaned = processor._clean_json_response(dirty_json)
        
        assert '```json' not in cleaned
        assert '```' not in cleaned
        assert '"invoice_id": "INV-001"' in cleaned
    
    def test_validate_extracted_data_success(self, sample_llm_response):
        """Test successful data validation"""
        processor = LLMProcessor("test-model")
        
        data = json.loads(sample_llm_response)
        result = processor._validate_extracted_data(data)
        
        assert isinstance(result, dict)
        assert result['invoice_id'] == "INV-2024-001"
    
    def test_validate_extracted_data_failure(self, invalid_llm_response):
        """Test data validation failure with fallback"""
        processor = LLMProcessor("test-model")
        
        data = json.loads(invalid_llm_response)
        result = processor._validate_extracted_data(data)
        
        # Should clean the data even if validation fails
        assert isinstance(result, dict)
        assert 'invoice_id' in result
        assert 'supplier' in result
    
    def test_clean_extracted_data(self):
        """Test data cleaning"""
        processor = LLMProcessor("test-model")
        
        dirty_data = {
            'invoice_id': 'INV-001',
            'issue_date': '15.01.2024',  # Non-ISO format
            'total_incl_vat': '€123.45',  # String with currency
            'supplier': {
                'name': 'Test Company'
            }
        }
        
        cleaned = processor._clean_extracted_data(dirty_data)
        
        assert cleaned['invoice_id'] == 'INV-001'
        assert cleaned['issue_date'] == '2024-01-15'  # Should be converted to ISO
        assert cleaned['total_incl_vat'] == 123.45    # Should be converted to float
        assert cleaned['supplier']['name'] == 'Test Company'
    
    def test_clean_date(self):
        """Test date cleaning and formatting"""
        processor = LLMProcessor("test-model")
        
        # Test various date formats
        assert processor._clean_date('2024-01-15') == '2024-01-15'
        assert processor._clean_date('15.01.2024') == '2024-01-15'
        assert processor._clean_date('15/01/2024') == '2024-01-15'
        assert processor._clean_date('15-01-2024') == '2024-01-15'
        
        # Test invalid date
        result = processor._clean_date('invalid-date')
        assert len(result) == 10  # Should return current date in YYYY-MM-DD format
    
    def test_clean_amount(self):
        """Test amount cleaning"""
        processor = LLMProcessor("test-model")
        
        # Test various amount formats
        assert processor._clean_amount(123.45) == 123.45
        assert processor._clean_amount('123.45') == 123.45
        assert processor._clean_amount('€123.45') == 123.45
        assert processor._clean_amount('1,234.56') == 1234.56
        assert processor._clean_amount('1.234,56') == 1234.56
        assert processor._clean_amount('invalid') == 0.0
        assert processor._clean_amount(None) == 0.0
    
    def test_clean_line_item(self):
        """Test line item cleaning"""
        processor = LLMProcessor("test-model")
        
        dirty_item = {
            'description': 'Software License',
            'quantity': '2',
            'unit_price': '€50.00',
            'line_total': '100,00',
            'vat_rate': '19%'
        }
        
        cleaned = processor._clean_line_item(dirty_item)
        
        assert cleaned['description'] == 'Software License'
        assert cleaned['quantity'] == 2.0
        assert cleaned['unit_price'] == 50.0
        assert cleaned['line_total'] == 100.0
        assert cleaned['vat_rate'] == 0.19
    
    def test_create_fallback_response(self, sample_invoice_text):
        """Test fallback response creation"""
        processor = LLMProcessor("test-model")
        
        result = processor._create_fallback_response(sample_invoice_text)
        
        assert isinstance(result, dict)
        assert 'invoice_id' in result
        assert 'issue_date' in result
        assert 'currency_code' in result
        assert 'total_incl_vat' in result
        assert 'raw_text' in result
        assert len(result['raw_text']) <= 1000  # Should be truncated
    
    def test_extract_invoice_number(self, sample_invoice_text):
        """Test invoice number extraction"""
        processor = LLMProcessor("test-model")
        
        invoice_number = processor._extract_invoice_number(sample_invoice_text)
        
        assert invoice_number == "INV-2024-001"
    
    def test_extract_date(self, sample_invoice_text):
        """Test date extraction"""
        processor = LLMProcessor("test-model")
        
        date = processor._extract_date(sample_invoice_text)
        
        assert date == "2024-01-15"
    
    def test_extract_currency(self, sample_invoice_text):
        """Test currency extraction"""
        processor = LLMProcessor("test-model")
        
        # Test with EUR text
        currency = processor._extract_currency(sample_invoice_text)
        assert currency == "EUR"
        
        # Test with dollar sign
        usd_text = "Invoice total: $123.45"
        currency = processor._extract_currency(usd_text)
        assert currency == "USD"
        
        # Test with pound sign
        gbp_text = "Invoice total: £123.45"
        currency = processor._extract_currency(gbp_text)
        assert currency == "GBP"
    
    def test_extract_total_amount(self, sample_invoice_text):
        """Test total amount extraction"""
        processor = LLMProcessor("test-model")
        
        total = processor._extract_total_amount(sample_invoice_text)
        
        assert total == 178.50
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup method"""
        with patch('src.core.llm_processor.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = Mock()
            
            processor = LLMProcessor("test-model")
            processor.model = Mock()
            processor.model.cleanup = Mock()
            
            await processor.cleanup()
            
            processor.model.cleanup.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_vllm_error(self, sample_invoice_text):
        """Test vLLM generation error handling"""
        with patch('src.core.llm_processor.VLLM_AVAILABLE', True):
            processor = LLMProcessor("mistral-7b-instruct", use_vllm=True)
            processor.model = Mock()
            processor.sampling_params = Mock()
            
            # Mock generation failure
            processor.model.generate.side_effect = Exception("Generation failed")
            
            with pytest.raises(Exception):
                await processor._generate_vllm("test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_transformers_error(self, sample_invoice_text):
        """Test Transformers generation error handling"""
        with patch('src.core.llm_processor.TRANSFORMERS_AVAILABLE', True):
            processor = LLMProcessor("mistral-7b-instruct", use_vllm=False)
            processor.model = Mock()
            processor.tokenizer = Mock()
            
            # Mock tokenization failure
            processor.tokenizer.side_effect = Exception("Tokenization failed")
            
            with pytest.raises(Exception):
                await processor._generate_transformers("test prompt")
    
    @pytest.mark.parametrize("language", ["en", "de", "et"])
    async def test_extract_structured_data_different_languages(self, sample_invoice_text, sample_llm_response, language):
        """Test structured data extraction with different languages"""
        with patch('src.core.llm_processor.VLLM_AVAILABLE', True):
            processor = LLMProcessor("mistral-7b-instruct", use_vllm=True)
            processor.model = Mock()
            processor.sampling_params = Mock()
            
            # Mock generation
            mock_output = Mock()
            mock_output.outputs = [Mock()]
            mock_output.outputs[0].text = sample_llm_response
            processor.model.generate.return_value = [mock_output]
            
            result = await processor.extract_structured_data(sample_invoice_text, language)
            
            assert isinstance(result, dict)
            assert result['invoice_id'] == "INV-2024-001"
    
    def test_model_name_mapping(self):
        """Test model name mapping for different models"""
        processor = LLMProcessor("mistral-7b-instruct")
        
        # The processor should handle model name mapping
        assert processor.model_name == "mistral-7b-instruct"
        
        # Test other model names
        processor2 = LLMProcessor("qwen2.5-7b")
        assert processor2.model_name == "qwen2.5-7b"
        
        processor3 = LLMProcessor("llama-3.1-8b")
        assert processor3.model_name == "llama-3.1-8b"
    
    def test_memory_efficiency(self):
        """Test memory efficiency considerations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create processor instance
        processor = LLMProcessor("test-model")
        
        # Test large text processing
        large_text = "This is a test. " * 1000
        fallback = processor._create_fallback_response(large_text)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert isinstance(fallback, dict)
        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase
    
    @pytest.mark.slow
    async def test_performance_large_text(self, sample_llm_response):
        """Test performance with large input text"""
        with patch('src.core.llm_processor.VLLM_AVAILABLE', True):
            processor = LLMProcessor("mistral-7b-instruct", use_vllm=True)
            processor.model = Mock()
            processor.sampling_params = Mock()
            
            # Mock generation
            mock_output = Mock()
            mock_output.outputs = [Mock()]
            mock_output.outputs[0].text = sample_llm_response
            processor.model.generate.return_value = [mock_output]
            
            # Create large text
            large_text = "INVOICE DATA " * 1000
            
            import time
            start_time = time.time()
            result = await processor.extract_structured_data(large_text, "en")
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            assert isinstance(result, dict)
            assert processing_time < 5.0  # Should complete within 5 seconds for mock
