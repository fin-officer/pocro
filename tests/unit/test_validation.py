"""
Unit tests for invoice data validation
"""
import pytest
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, Any

from src.models.validation import InvoiceValidator, validate_invoice_data, quick_validate
from src.models.invoice_schema import InvoiceValidationResult, ValidationError


class TestInvoiceValidator:
    """Test cases for InvoiceValidator class"""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return InvoiceValidator()
    
    @pytest.fixture
    def valid_invoice_data(self):
        """Valid invoice data for testing"""
        return {
            "invoice_id": "INV-2024-001",
            "issue_date": "2024-01-15",
            "currency_code": "EUR",
            "supplier": {
                "name": "Test Company GmbH",
                "vat_id": "DE123456789",
                "country_code": "DE",
                "address_line": "Test Street 123",
                "city": "Test City",
                "postal_code": "12345"
            },
            "customer": {
                "name": "Customer Ltd",
                "vat_id": "EE123456789",
                "country_code": "EE",
                "address_line": "Customer Road 456",
                "city": "Customer City",
                "postal_code": "54321"
            },
            "invoice_lines": [
                {
                    "line_id": "1",
                    "description": "Software License",
                    "quantity": 1.0,
                    "unit_price": 100.00,
                    "line_total": 100.00,
                    "vat_rate": 0.19
                }
            ],
            "total_excl_vat": 100.00,
            "total_vat": 19.00,
            "total_incl_vat": 119.00,
            "payable_amount": 119.00
        }
    
    @pytest.fixture
    def invalid_invoice_data(self):
        """Invalid invoice data for testing"""
        return {
            "invoice_id": "",  # Invalid: empty
            "issue_date": "invalid-date",  # Invalid: bad format
            "currency_code": "INVALID",  # Invalid: not supported
            "supplier": {
                "name": "",  # Invalid: empty
                "vat_id": "INVALID123",  # Invalid: bad format
                "country_code": "XX"  # Invalid: not supported
            },
            "customer": {
                "name": "Customer Ltd",
                "vat_id": "INVALID456",  # Invalid: bad format
                "country_code": "YY"  # Invalid: not supported
            },
            "total_excl_vat": -100,  # Invalid: negative
            "total_vat": "not_a_number",  # Invalid: not numeric
            "total_incl_vat": 50  # Invalid: doesn't match calculation
        }
    
    def test_validator_initialization(self, validator):
        """Test validator initialization"""
        assert isinstance(validator.vat_patterns, dict)
        assert isinstance(validator.country_currencies, dict)
        assert 'DE' in validator.vat_patterns
        assert 'DE' in validator.country_currencies
    
    def test_validate_invoice_valid_data(self, validator, valid_invoice_data):
        """Test validation with valid data"""
        result = validator.validate_invoice(valid_invoice_data)
        
        assert isinstance(result, InvoiceValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.completeness_score > 0.8
        assert result.confidence_score > 0.8
    
    def test_validate_invoice_invalid_data(self, validator, invalid_invoice_data):
        """Test validation with invalid data"""
        result = validator.validate_invoice(invalid_invoice_data)
        
        assert isinstance(result, InvoiceValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
        
        # Check that specific errors are detected
        error_fields = [error.field for error in result.errors]
        assert any('invoice_id' in field for field in error_fields)
        assert any('issue_date' in field for field in error_fields)
    
    def test_validate_invoice_number_formats(self, validator):
        """Test invoice number format validation"""
        valid_numbers = [
            "INV-2024-001",
            "12345",
            "ABC123456",
            "2024/0001",
            "RE/240001"
        ]
        
        for number in valid_numbers:
            assert validator._validate_invoice_number(number) is True
        
        invalid_numbers = [
            "",
            "12",
            "AB",
            None
        ]
        
        for number in invalid_numbers:
            assert validator._validate_invoice_number(number) is False
    
    def test_validate_vat_id_formats(self, validator):
        """Test VAT ID format validation"""
        # Test valid VAT IDs
        valid_vat_ids = {
            "DE": "DE123456789",
            "EE": "EE123456789",
            "GB": "GB123456789",
            "FR": "FRAA123456789",
            "IT": "IT12345678901"
        }
        
        for country, vat_id in valid_vat_ids.items():
            assert validator._validate_vat_id(vat_id, country) is True
        
        # Test invalid VAT IDs
        invalid_vat_ids = {
            "DE": "INVALID123",
            "EE": "EE12345",  # Too short
            "GB": "GB12345678901",  # Too long
            "FR": "FR123456789",  # Wrong format
        }
        
        for country, vat_id in invalid_vat_ids.items():
            assert validator._validate_vat_id(vat_id, country) is False
    
    def test_validate_financial_consistency(self, validator):
        """Test financial consistency validation"""
        # Test consistent data
        consistent_data = {
            "total_excl_vat": 100.00,
            "total_vat": 19.00,
            "total_incl_vat": 119.00,
            "invoice_lines": [
                {
                    "line_total": 100.00,
                    "quantity": 1.0,
                    "unit_price": 100.00
                }
            ]
        }
        
        errors = validator._validate_financial_consistency(consistent_data)
        assert len(errors) == 0
        
        # Test inconsistent data
        inconsistent_data = {
            "total_excl_vat": 100.00,
            "total_vat": 19.00,
            "total_incl_vat": 200.00,  # Inconsistent
            "invoice_lines": [
                {
                    "line_total": 50.00,  # Doesn't match total_excl_vat
                    "quantity": 1.0,
                    "unit_price": 50.00
                }
            ]
        }
        
        errors = validator._validate_financial_consistency(inconsistent_data)
        assert len(errors) > 0
    
    def test_validate_dates(self, validator):
        """Test date validation"""
        # Test valid dates
        valid_data = {
            "issue_date": "2024-01-15",
            "payment_terms": {
                "payment_due_date": "2024-02-15"
            }
        }
        
        errors = validator._validate_dates(valid_data)
        assert len(errors) == 0
        
        # Test invalid dates
        invalid_data = {
            "issue_date": "invalid-date",
            "payment_terms": {
                "payment_due_date": "2024-01-01"  # Before issue date
            }
        }
        
        errors = validator._validate_dates(invalid_data)
        assert len(errors) > 0
        
        # Test future date
        future_data = {
            "issue_date": "2030-01-01"  # Future date
        }
        
        errors = validator._validate_dates(future_data)
        assert len(errors) > 0
    
    def test_validate_line_items(self, validator):
        """Test line item validation"""
        # Test valid line items
        valid_data = {
            "invoice_lines": [
                {
                    "description": "Software License",
                    "quantity": 2.0,
                    "unit_price": 50.00,
                    "line_total": 100.00
                }
            ]
        }
        
        errors = validator._validate_line_items(valid_data)
        assert len(errors) == 0
        
        # Test invalid line items
        invalid_data = {
            "invoice_lines": [
                {
                    "description": "",  # Empty description
                    "quantity": 2.0,
                    "unit_price": 50.00,
                    "line_total": 200.00  # Inconsistent calculation
                }
            ]
        }
        
        errors = validator._validate_line_items(invalid_data)
        assert len(errors) > 0
    
    def test_calculate_completeness_score(self, validator, valid_invoice_data):
        """Test completeness score calculation"""
        score = validator._calculate_completeness_score(valid_invoice_data)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high for complete data
        
        # Test with incomplete data
        incomplete_data = {
            "invoice_id": "INV-001",
            "issue_date": "2024-01-15"
        }
        
        incomplete_score = validator._calculate_completeness_score(incomplete_data)
        assert incomplete_score < score
    
    def test_calculate_confidence_score(self, validator, valid_invoice_data):
        """Test confidence score calculation"""
        score = validator._calculate_confidence_score(valid_invoice_data)
        
        assert 0.0 <= score <= 1.0
        
        # Test with processing metadata
        data_with_metadata = valid_invoice_data.copy()
        data_with_metadata['processing_metadata'] = {
            'ocr_confidence': 0.95
        }
        
        score_with_metadata = validator._calculate_confidence_score(data_with_metadata)
        assert score_with_metadata >= score
    
    def test_calculate_consistency_score(self, validator, valid_invoice_data):
        """Test consistency score calculation"""
        score = validator._calculate_consistency_score(valid_invoice_data)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high for consistent data
        
        # Test with inconsistent data
        inconsistent_data = valid_invoice_data.copy()
        inconsistent_data['total_incl_vat'] = 500.00  # Very inconsistent
        
        inconsistent_score = validator._calculate_consistency_score(inconsistent_data)
        assert inconsistent_score < score
    
    def test_safe_decimal_conversion(self, validator):
        """Test safe decimal conversion"""
        # Test various input types
        assert validator._safe_decimal_conversion(123) == Decimal('123')
        assert validator._safe_decimal_conversion(123.45) == Decimal('123.45')
        assert validator._safe_decimal_conversion("123.45") == Decimal('123.45')
        assert validator._safe_decimal_conversion("€123.45") == Decimal('123.45')
        assert validator._safe_decimal_conversion("1,234.56") == Decimal('1234.56')
        assert validator._safe_decimal_conversion("1.234,56") == Decimal('1234.56')
        assert validator._safe_decimal_conversion("invalid") == Decimal('0')
        assert validator._safe_decimal_conversion(None) == Decimal('0')
    
    def test_parse_date(self, validator):
        """Test date parsing"""
        # Test various date formats
        assert validator._parse_date("2024-01-15") == date(2024, 1, 15)
        assert validator._parse_date("15.01.2024") == date(2024, 1, 15)
        assert validator._parse_date("15/01/2024") == date(2024, 1, 15)
        assert validator._parse_date("15-01-2024") == date(2024, 1, 15)
        assert validator._parse_date("invalid-date") is None
        assert validator._parse_date("") is None
    
    def test_get_nested_value(self, validator, valid_invoice_data):
        """Test nested value extraction"""
        # Test existing nested value
        value = validator._get_nested_value(valid_invoice_data, "supplier.name")
        assert value == "Test Company GmbH"
        
        # Test non-existing nested value
        value = validator._get_nested_value(valid_invoice_data, "supplier.nonexistent")
        assert value is None
        
        # Test empty value
        empty_data = {"supplier": {"name": ""}}
        value = validator._get_nested_value(empty_data, "supplier.name")
        assert value is None
    
    def test_extract_pydantic_errors(self, validator):
        """Test Pydantic error extraction"""
        # Mock a Pydantic validation error
        class MockPydanticError(Exception):
            def errors(self):
                return [
                    {
                        'loc': ('invoice_id',),
                        'msg': 'field required',
                        'input': None
                    },
                    {
                        'loc': ('supplier', 'vat_id'),
                        'msg': 'invalid format',
                        'input': 'INVALID'
                    }
                ]
        
        mock_error = MockPydanticError()
        errors = validator._extract_pydantic_errors(mock_error)
        
        assert len(errors) == 2
        assert errors[0].field == 'invoice_id'
        assert errors[0].message == 'field required'
        assert errors[1].field == 'supplier.vat_id'
        assert errors[1].message == 'invalid format'
    
    def test_validate_business_warnings(self, validator, valid_invoice_data):
        """Test business warning generation"""
        warnings = validator._validate_business_warnings(valid_invoice_data)
        
        assert isinstance(warnings, list)
        # May or may not have warnings depending on data
        
        # Test with unusual currency
        unusual_currency_data = valid_invoice_data.copy()
        unusual_currency_data['currency_code'] = 'USD'  # Unusual for DE supplier
        
        warnings = validator._validate_business_warnings(unusual_currency_data)
        assert any('currency' in warning.lower() for warning in warnings)
        
        # Test with very high amount
        high_amount_data = valid_invoice_data.copy()
        high_amount_data['total_incl_vat'] = 200000.00
        
        warnings = validator._validate_business_warnings(high_amount_data)
        assert any('high' in warning.lower() for warning in warnings)


class TestValidationHelperFunctions:
    """Test validation helper functions"""
    
    def test_validate_invoice_data_function(self, valid_invoice_data):
        """Test validate_invoice_data helper function"""
        result = validate_invoice_data(valid_invoice_data)
        
        assert isinstance(result, InvoiceValidationResult)
        assert result.is_valid is True
    
    def test_quick_validate_function(self, valid_invoice_data, invalid_invoice_data):
        """Test quick_validate helper function"""
        # Test valid data
        is_valid, errors = quick_validate(valid_invoice_data)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test invalid data
        is_valid, errors = quick_validate(invalid_invoice_data)
        assert is_valid is False
        assert len(errors) > 0
        assert all(isinstance(error, str) for error in errors)


class TestValidationErrorHandling:
    """Test validation error handling"""
    
    def test_validation_with_none_input(self):
        """Test validation with None input"""
        result = validate_invoice_data(None)
        
        assert isinstance(result, InvoiceValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validation_with_empty_dict(self):
        """Test validation with empty dictionary"""
        result = validate_invoice_data({})
        
        assert isinstance(result, InvoiceValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validation_with_malformed_data(self):
        """Test validation with malformed data"""
        malformed_data = {
            "invoice_id": 123,  # Should be string
            "supplier": "not a dict",  # Should be dict
            "invoice_lines": "not a list"  # Should be list
        }
        
        result = validate_invoice_data(malformed_data)
        
        assert isinstance(result, InvoiceValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0


class TestValidationPerformance:
    """Test validation performance"""
    
    def test_validation_performance_large_dataset(self, valid_invoice_data):
        """Test validation performance with large line items"""
        # Create data with many line items
        large_data = valid_invoice_data.copy()
        large_data['invoice_lines'] = []
        
        for i in range(100):
            large_data['invoice_lines'].append({
                "line_id": str(i + 1),
                "description": f"Item {i + 1}",
                "quantity": 1.0,
                "unit_price": 10.00,
                "line_total": 10.00,
                "vat_rate": 0.19
            })
        
        # Update totals
        large_data['total_excl_vat'] = 1000.00
        large_data['total_vat'] = 190.00
        large_data['total_incl_vat'] = 1190.00
        
        import time
        start_time = time.time()
        result = validate_invoice_data(large_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert isinstance(result, InvoiceValidationResult)
        assert processing_time < 5.0  # Should complete within 5 seconds
    
    def test_validation_memory_efficiency(self, valid_invoice_data):
        """Test validation memory efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Validate multiple times
        for _ in range(100):
            result = validate_invoice_data(valid_invoice_data)
            assert isinstance(result, InvoiceValidationResult)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be minimal
        assert memory_increase < 50  # Less than 50MB increase


class TestValidationEdgeCases:
    """Test validation edge cases"""
    
    def test_validation_with_extreme_values(self):
        """Test validation with extreme values"""
        extreme_data = {
            "invoice_id": "A" * 1000,  # Very long ID
            "issue_date": "2024-01-15",
            "currency_code": "EUR",
            "supplier": {
                "name": "Test Company",
                "country_code": "DE"
            },
            "customer": {
                "name": "Customer",
                "country_code": "EE"
            },
            "total_excl_vat": 99999999.99,  # Very large amount
            "total_vat": 19999999.99,
            "total_incl_vat": 119999999.98
        }
        
        result = validate_invoice_data(extreme_data)
        
        assert isinstance(result, InvoiceValidationResult)
        # Should handle extreme values gracefully
