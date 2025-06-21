"""
Invoice data validation utilities
"""

import logging
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from .invoice_schema import InvoiceData, InvoiceValidationResult, ValidationError, VATCategory

logger = logging.getLogger(__name__)


class InvoiceValidator:
    """Comprehensive invoice data validator"""

    def __init__(self):
        self.vat_patterns = {
            "DE": r"^DE[0-9]{9}$",  # Germany
            "EE": r"^EE[0-9]{9}$",  # Estonia
            "GB": r"^GB[0-9]{9}$",  # United Kingdom
            "FR": r"^FR[0-9A-Z]{2}[0-9]{9}$",  # France
            "IT": r"^IT[0-9]{11}$",  # Italy
            "ES": r"^ES[0-9A-Z][0-9]{7}[0-9A-Z]$",  # Spain
            "NL": r"^NL[0-9]{9}B[0-9]{2}$",  # Netherlands
            "AT": r"^ATU[0-9]{8}$",  # Austria
            "BE": r"^BE[0-9]{10}$",  # Belgium
            "PL": r"^PL[0-9]{10}$",  # Poland
            "FI": r"^FI[0-9]{8}$",  # Finland
            "SE": r"^SE[0-9]{12}$",  # Sweden
            "DK": r"^DK[0-9]{8}$",  # Denmark
        }

        self.country_currencies = {
            "DE": "EUR",
            "EE": "EUR",
            "FR": "EUR",
            "IT": "EUR",
            "ES": "EUR",
            "NL": "EUR",
            "AT": "EUR",
            "BE": "EUR",
            "FI": "EUR",
            "GB": "GBP",
            "PL": "PLN",
            "SE": "SEK",
            "DK": "DKK",
        }

    def validate_invoice(self, invoice_data: Dict[str, Any]) -> InvoiceValidationResult:
        """
        Comprehensive validation of invoice data

        Args:
            invoice_data: Raw invoice data dictionary

        Returns:
            Validation result with errors and scores
        """
        errors = []
        warnings = []

        try:
            # Try to create InvoiceData object for Pydantic validation
            validated_invoice = InvoiceData(**invoice_data)
            logger.debug("Pydantic validation passed")

        except Exception as e:
            # Collect Pydantic validation errors
            errors.extend(self._extract_pydantic_errors(e))
            validated_invoice = None

        # Additional business logic validation
        errors.extend(self._validate_business_rules(invoice_data))
        warnings.extend(self._validate_business_warnings(invoice_data))

        # Calculate scores
        completeness_score = self._calculate_completeness_score(invoice_data)
        confidence_score = self._calculate_confidence_score(invoice_data)

        return InvoiceValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            completeness_score=completeness_score,
            confidence_score=confidence_score,
        )

    def _extract_pydantic_errors(self, exception: Exception) -> List[ValidationError]:
        """Extract validation errors from Pydantic exception"""
        errors = []

        if hasattr(exception, "errors"):
            for error in exception.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                errors.append(ValidationError(field=field_path, message=error["msg"], value=error.get("input")))
        else:
            errors.append(ValidationError(field="general", message=str(exception), value=None))

        return errors

    def _validate_business_rules(self, invoice_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate business-specific rules"""
        errors = []

        # Validate invoice number format
        invoice_id = invoice_data.get("invoice_id", "")
        if not self._validate_invoice_number(invoice_id):
            errors.append(
                ValidationError(field="invoice_id", message="Invoice number format is invalid", value=invoice_id)
            )

        # Validate VAT IDs
        supplier = invoice_data.get("supplier", {})
        if supplier.get("vat_id"):
            if not self._validate_vat_id(supplier["vat_id"], supplier.get("country_code")):
                errors.append(
                    ValidationError(
                        field="supplier.vat_id", message="Supplier VAT ID format is invalid", value=supplier["vat_id"]
                    )
                )

        customer = invoice_data.get("customer", {})
        if customer.get("vat_id"):
            if not self._validate_vat_id(customer["vat_id"], customer.get("country_code")):
                errors.append(
                    ValidationError(
                        field="customer.vat_id", message="Customer VAT ID format is invalid", value=customer["vat_id"]
                    )
                )

        # Validate financial consistency
        errors.extend(self._validate_financial_consistency(invoice_data))

        # Validate date logic
        errors.extend(self._validate_dates(invoice_data))

        # Validate line items
        errors.extend(self._validate_line_items(invoice_data))

        return errors

    def _validate_business_warnings(self, invoice_data: Dict[str, Any]) -> List[str]:
        """Generate business logic warnings"""
        warnings = []

        # Check currency consistency
        supplier_country = invoice_data.get("supplier", {}).get("country_code")
        currency = invoice_data.get("currency_code")

        if supplier_country and currency:
            expected_currency = self.country_currencies.get(supplier_country)
            if expected_currency and expected_currency != currency:
                warnings.append(
                    f"Currency {currency} is unusual for country {supplier_country}, " f"expected {expected_currency}"
                )

        # Check for missing optional but important fields
        if not invoice_data.get("customer", {}).get("vat_id"):
            warnings.append("Customer VAT ID is missing")

        if not invoice_data.get("purchase_order_reference"):
            warnings.append("Purchase order reference is missing")

        # Check for unusually high amounts
        total = self._safe_decimal_conversion(invoice_data.get("total_incl_vat", 0))
        if total > Decimal("100000"):
            warnings.append(f"Invoice total {total} is unusually high")

        return warnings

    def _validate_invoice_number(self, invoice_id: str) -> bool:
        """Validate invoice number format"""
        if not invoice_id or len(invoice_id) < 3:
            return False

        # Common invoice number patterns
        patterns = [
            r"^[A-Z]{1,4}-\d{4,}$",  # ABC-1234
            r"^\d{4,}$",  # 1234567
            r"^[A-Z]{1,4}\d{4,}$",  # ABC1234
            r"^\d{4}/\d{4}$",  # 2024/0001
            r"^[A-Z]{1,4}/\d{4,}$",  # INV/2024001
        ]

        return any(re.match(pattern, invoice_id) for pattern in patterns)

    def _validate_vat_id(self, vat_id: str, country_code: str) -> bool:
        """Validate VAT ID format for specific country"""
        if not vat_id or not country_code:
            return False

        pattern = self.vat_patterns.get(country_code.upper())
        if not pattern:
            return True  # Unknown country, skip validation

        return bool(re.match(pattern, vat_id.upper()))

    def _validate_financial_consistency(self, invoice_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate financial calculations consistency"""
        errors = []

        try:
            total_excl_vat = self._safe_decimal_conversion(invoice_data.get("total_excl_vat", 0))
            total_vat = self._safe_decimal_conversion(invoice_data.get("total_vat", 0))
            total_incl_vat = self._safe_decimal_conversion(invoice_data.get("total_incl_vat", 0))

            # Check if totals are consistent
            calculated_total = total_excl_vat + total_vat
            if abs(calculated_total - total_incl_vat) > Decimal("0.01"):
                errors.append(
                    ValidationError(
                        field="total_incl_vat",
                        message=f"Total including VAT {total_incl_vat} does not match "
                        f"sum of net {total_excl_vat} and VAT {total_vat} = {calculated_total}",
                        value=float(total_incl_vat),
                    )
                )

            # Validate line items sum
            invoice_lines = invoice_data.get("invoice_lines", [])
            if invoice_lines:
                line_total_sum = sum(self._safe_decimal_conversion(line.get("line_total", 0)) for line in invoice_lines)

                if abs(line_total_sum - total_excl_vat) > Decimal("0.01"):
                    errors.append(
                        ValidationError(
                            field="total_excl_vat",
                            message=f"Total excluding VAT {total_excl_vat} does not match "
                            f"sum of line totals {line_total_sum}",
                            value=float(total_excl_vat),
                        )
                    )

        except Exception as e:
            errors.append(
                ValidationError(
                    field="financial_totals", message=f"Error validating financial consistency: {str(e)}", value=None
                )
            )

        return errors

    def _validate_dates(self, invoice_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate date fields"""
        errors = []

        # Validate issue date
        issue_date = invoice_data.get("issue_date")
        if issue_date:
            parsed_date = self._parse_date(issue_date)
            if not parsed_date:
                errors.append(ValidationError(field="issue_date", message="Invalid date format", value=issue_date))
            elif parsed_date > date.today():
                errors.append(
                    ValidationError(field="issue_date", message="Issue date cannot be in the future", value=issue_date)
                )

        # Validate payment due date if present
        payment_terms = invoice_data.get("payment_terms", {})
        if payment_terms and payment_terms.get("payment_due_date"):
            due_date = self._parse_date(payment_terms["payment_due_date"])
            if not due_date:
                errors.append(
                    ValidationError(
                        field="payment_terms.payment_due_date",
                        message="Invalid payment due date format",
                        value=payment_terms["payment_due_date"],
                    )
                )
            elif issue_date and due_date:
                issue_parsed = self._parse_date(issue_date)
                if issue_parsed and due_date < issue_parsed:
                    errors.append(
                        ValidationError(
                            field="payment_terms.payment_due_date",
                            message="Payment due date cannot be before issue date",
                            value=payment_terms["payment_due_date"],
                        )
                    )

        return errors

    def _validate_line_items(self, invoice_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate individual line items"""
        errors = []

        invoice_lines = invoice_data.get("invoice_lines", [])

        for i, line in enumerate(invoice_lines):
            if not isinstance(line, dict):
                errors.append(
                    ValidationError(field=f"invoice_lines.{i}", message="Line item must be an object", value=line)
                )
                continue

            # Validate line calculations
            try:
                quantity = self._safe_decimal_conversion(line.get("quantity", 0))
                unit_price = self._safe_decimal_conversion(line.get("unit_price", 0))
                line_total = self._safe_decimal_conversion(line.get("line_total", 0))

                calculated_total = quantity * unit_price
                if abs(calculated_total - line_total) > Decimal("0.01"):
                    errors.append(
                        ValidationError(
                            field=f"invoice_lines.{i}.line_total",
                            message=f"Line total {line_total} does not match quantity {quantity} × unit price {unit_price} = {calculated_total}",
                            value=float(line_total),
                        )
                    )

            except Exception as e:
                errors.append(
                    ValidationError(
                        field=f"invoice_lines.{i}",
                        message=f"Error validating line item calculations: {str(e)}",
                        value=line,
                    )
                )

            # Validate required fields
            if not line.get("description", "").strip():
                errors.append(
                    ValidationError(
                        field=f"invoice_lines.{i}.description",
                        message="Line item description is required",
                        value=line.get("description"),
                    )
                )

        return errors

    def _calculate_completeness_score(self, invoice_data: Dict[str, Any]) -> float:
        """Calculate data completeness score (0-1)"""
        required_fields = [
            "invoice_id",
            "issue_date",
            "supplier.name",
            "customer.name",
            "total_incl_vat",
            "currency_code",
        ]

        optional_important_fields = [
            "supplier.vat_id",
            "customer.vat_id",
            "supplier.country_code",
            "customer.country_code",
            "invoice_lines",
            "total_excl_vat",
            "total_vat",
        ]

        total_fields = len(required_fields) + len(optional_important_fields)
        present_fields = 0

        # Check required fields
        for field in required_fields:
            if self._get_nested_value(invoice_data, field):
                present_fields += 1

        # Check optional fields (half weight)
        for field in optional_important_fields:
            if self._get_nested_value(invoice_data, field):
                present_fields += 0.5

        return min(present_fields / total_fields, 1.0)

    def _calculate_confidence_score(self, invoice_data: Dict[str, Any]) -> float:
        """Calculate extraction confidence score (0-1)"""
        score = 0.0
        factors = 0

        # OCR confidence if available
        ocr_confidence = invoice_data.get("processing_metadata", {}).get("ocr_confidence")
        if ocr_confidence is not None:
            score += float(ocr_confidence)
            factors += 1

        # Data consistency factor
        consistency_score = self._calculate_consistency_score(invoice_data)
        score += consistency_score
        factors += 1

        # Completeness factor
        completeness = self._calculate_completeness_score(invoice_data)
        score += completeness
        factors += 1

        return score / factors if factors > 0 else 0.0

    def _calculate_consistency_score(self, invoice_data: Dict[str, Any]) -> float:
        """Calculate internal data consistency score"""
        score = 1.0

        # Check financial consistency
        try:
            total_excl_vat = self._safe_decimal_conversion(invoice_data.get("total_excl_vat", 0))
            total_vat = self._safe_decimal_conversion(invoice_data.get("total_vat", 0))
            total_incl_vat = self._safe_decimal_conversion(invoice_data.get("total_incl_vat", 0))

            if total_incl_vat > 0:
                calculated_total = total_excl_vat + total_vat
                difference = abs(calculated_total - total_incl_vat) / total_incl_vat
                score -= min(difference * 2, 0.3)  # Max penalty 0.3
        except:
            score -= 0.2

        # Check line items consistency
        invoice_lines = invoice_data.get("invoice_lines", [])
        if invoice_lines:
            try:
                for line in invoice_lines:
                    quantity = self._safe_decimal_conversion(line.get("quantity", 0))
                    unit_price = self._safe_decimal_conversion(line.get("unit_price", 0))
                    line_total = self._safe_decimal_conversion(line.get("line_total", 0))

                    if line_total > 0:
                        calculated = quantity * unit_price
                        difference = abs(calculated - line_total) / line_total
                        score -= min(difference * 0.1, 0.05)  # Small penalty per line
            except:
                score -= 0.1

        return max(score, 0.0)

    def _safe_decimal_conversion(self, value: Any) -> Decimal:
        """Safely convert value to Decimal"""
        if isinstance(value, Decimal):
            return value

        if isinstance(value, (int, float)):
            return Decimal(str(value))

        if isinstance(value, str):
            # Clean string value
            cleaned = re.sub(r"[^\d.,\-]", "", value)
            if not cleaned:
                return Decimal("0")

            # Handle different decimal separators
            if "," in cleaned and "." in cleaned:
                # European format: 1.234,56
                cleaned = cleaned.replace(".", "").replace(",", ".")
            elif "," in cleaned:
                # Check if comma is decimal separator
                parts = cleaned.split(",")
                if len(parts) == 2 and len(parts[1]) <= 2:
                    cleaned = cleaned.replace(",", ".")
                else:
                    cleaned = cleaned.replace(",", "")

            try:
                return Decimal(cleaned)
            except InvalidOperation:
                return Decimal("0")

        return Decimal("0")

    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string in various formats"""
        if not date_str:
            return None

        formats = ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        return None

    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested dictionary value using dot notation"""
        keys = field_path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value if value != "" else None


def validate_invoice_data(invoice_data: Dict[str, Any]) -> InvoiceValidationResult:
    """
    Convenience function for invoice validation

    Args:
        invoice_data: Invoice data dictionary

    Returns:
        Validation result
    """
    validator = InvoiceValidator()
    return validator.validate_invoice(invoice_data)


def quick_validate(invoice_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Quick validation returning simple boolean and error messages

    Args:
        invoice_data: Invoice data dictionary

    Returns:
        Tuple of (is_valid, error_messages)
    """
    result = validate_invoice_data(invoice_data)
    error_messages = [f"{error.field}: {error.message}" for error in result.errors]

    return result.is_valid, error_messages
