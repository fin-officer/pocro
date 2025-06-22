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

    def validate_invoice(self, invoice_data: Optional[Dict[str, Any]]) -> InvoiceValidationResult:
        """
        Comprehensive validation of invoice data

        Args:
            invoice_data: Raw invoice data dictionary or None

        Returns:
            Validation result with errors and scores
        """
        errors = []
        warnings = []

        if invoice_data is None:
            errors.append(ValidationError(field="invoice_data", message="Input data cannot be None", value=None))
            return InvoiceValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                completeness_score=0.0,
                confidence_score=0.0,
            )

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

        # Handle case where we get a string error about InvoiceItem not being subscriptable
        error_msg = str(exception)
        if "'InvoiceItem' object is not subscriptable" in error_msg:
            # This typically happens when trying to access dictionary-style on a model instance
            # We'll return a more helpful error message
            return [
                ValidationError(
                    field="invoice_lines", message="Error processing line items - invalid data structure", value=None
                )
            ]

        # Handle standard Pydantic validation errors
        if hasattr(exception, "errors"):
            for error in exception.errors():
                # Safely get location and message with fallbacks
                loc = error.get("loc", ())
                field_path = ".".join(str(loc) for loc in loc) if loc else "unknown_field"
                msg = error.get("msg", "Validation error occurred")

                # Handle specific error cases
                if "not a valid dict" in msg and "InvoiceItem" in msg:
                    msg = "Invalid line item format - expected a dictionary with required fields"

                errors.append(ValidationError(field=field_path, message=msg, value=error.get("input")))
        else:
            # For any other type of exception, include the full error message
            errors.append(ValidationError(field="general", message=f"Validation error: {error_msg}", value=None))

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
        if isinstance(supplier, dict) and supplier.get("vat_id"):
            if not self._validate_vat_id(
                supplier["vat_id"], supplier.get("country_code") if isinstance(supplier, dict) else None
            ):
                errors.append(
                    ValidationError(
                        field="supplier.vat_id", message="Supplier VAT ID format is invalid", value=supplier["vat_id"]
                    )
                )

        customer = invoice_data.get("customer", {})
        if isinstance(customer, dict) and customer.get("vat_id"):
            if not self._validate_vat_id(
                customer["vat_id"], customer.get("country_code") if isinstance(customer, dict) else None
            ):
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

    def _get_default_currency(self, country_code: str) -> Optional[str]:
        """Get the default currency for a given country code

        Args:
            country_code: ISO 2-letter country code

        Returns:
            Default currency code (3 letters) or None if not found
        """
        if not country_code or not isinstance(country_code, str):
            return None

        # Convert to uppercase for case-insensitive matching
        country_code = country_code.upper()

        # Return the currency code if the country is found, otherwise None
        return self.country_currencies.get(country_code)

    def _validate_business_warnings(self, invoice_data: Dict[str, Any]) -> List[str]:
        """Generate business logic warnings

        Args:
            invoice_data: The invoice data to validate

        Returns:
            List of warning messages
        """
        warnings = []

        # Safely get supplier and customer data with type checking
        supplier = invoice_data.get("supplier", {})
        customer = invoice_data.get("customer", {})

        # Only proceed with country/currency checks if supplier is a dict with country_code
        supplier_country = None
        if isinstance(supplier, dict):
            supplier_country = supplier.get("country_code")

        customer_country = None
        if isinstance(customer, dict):
            customer_country = customer.get("country_code")

        invoice_currency = invoice_data.get("currency_code")

        # Currency consistency check
        if supplier_country and invoice_currency:
            default_currency = self._get_default_currency(supplier_country)
            if default_currency and invoice_currency != default_currency:
                warnings.append(
                    f"Invoice currency {invoice_currency} differs from supplier's default currency {default_currency}"
                )

        # Check for missing fields
        required_fields = ["invoice_date", "due_date", "total_amount"]
        for field in required_fields:
            if field not in invoice_data:
                warnings.append(f"Missing recommended field: {field}")

        # Check for missing optional but important fields
        if not (isinstance(customer, dict) and customer.get("vat_id")):
            warnings.append("Customer VAT ID is missing")

        if not invoice_data.get("purchase_order_reference"):
            warnings.append("Purchase order reference is missing")

        # Check for unusually high amounts
        total = self._safe_decimal_conversion(invoice_data.get("total_incl_vat", 0))
        if total > Decimal("100000"):
            warnings.append(f"Invoice total {total} is unusually high")

        return warnings

    def _validate_invoice_number(self, invoice_id: Any) -> bool:
        """Validate invoice number format

        Args:
            invoice_id: The invoice ID to validate (can be any type, but should be a string)

        Returns:
            bool: True if the invoice ID is valid, False otherwise
        """
        if invoice_id is None:
            return False

        # Convert to string if it's not already
        try:
            invoice_str = str(invoice_id)
            return len(invoice_str) >= 3
        except (TypeError, ValueError):
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
            if not isinstance(line, (dict, object)):
                errors.append(
                    ValidationError(field=f"invoice_lines.{i}", message="Line item must be an object", value=line)
                )
                continue

            # Helper function to get attribute safely from either dict or object
            def get_attr(obj, attr, default=None):
                if isinstance(obj, dict):
                    return obj.get(attr, default)
                return getattr(obj, attr, default)

            # Validate line calculations
            try:
                quantity = self._safe_decimal_conversion(get_attr(line, "quantity", 0))
                unit_price = self._safe_decimal_conversion(get_attr(line, "unit_price", 0))
                line_total = self._safe_decimal_conversion(get_attr(line, "line_total", 0))

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
                        value=str(line) if hasattr(line, "__dict__") else line,
                    )
                )

            # Validate required fields
            description = get_attr(line, "description", "")
            if not (isinstance(description, str) and description.strip()):
                errors.append(
                    ValidationError(
                        field=f"invoice_lines.{i}.description",
                        message="Line item description is required",
                        value=description,
                    )
                )

        return errors

    def _calculate_completeness_score(self, invoice_data: Dict[str, Any]) -> float:
        """Calculate data completeness score (0-1)"""
        # Required fields (higher weight)
        required_fields = [
            "invoice_id",
            "issue_date",
            "supplier.name",
            "customer.name",
            "total_incl_vat",
            "currency_code",
            "supplier.country_code",  # Moved up as it's crucial for validation
            "customer.country_code",  # Moved up as it's crucial for validation
        ]

        # Important but optional fields (medium weight)
        important_fields = [
            "supplier.vat_id",
            "customer.vat_id",
            "supplier.address_line",
            "customer.address_line",
            "invoice_lines",
            "total_excl_vat",
            "total_vat",
            "payable_amount",
        ]

        # Less critical fields (lower weight)
        optional_fields = [
            "supplier.city",
            "supplier.postal_code",
            "customer.city",
            "customer.postal_code",
            "purchase_order_reference",
            "contract_reference",
        ]

        # Calculate weights
        required_weight = 1.0
        important_weight = 0.7
        optional_weight = 0.3

        total_score = 0.0
        max_score = 0.0

        # Check required fields
        for field in required_fields:
            max_score += required_weight
            if self._get_nested_value(invoice_data, field):
                total_score += required_weight

        # Check important fields
        for field in important_fields:
            max_score += important_weight
            if self._get_nested_value(invoice_data, field):
                total_score += important_weight

        # Check optional fields
        for field in optional_fields:
            max_score += optional_weight
            if self._get_nested_value(invoice_data, field):
                total_score += optional_weight

        # Ensure we don't divide by zero and return a score between 0 and 1
        if max_score == 0:
            return 0.0

        score = total_score / max_score
        return min(max(score, 0.0), 1.0)

    def _calculate_confidence_score(self, invoice_data: Dict[str, Any]) -> float:
        """Calculate extraction confidence score (0-1)"""
        # Calculate base score as average of consistency and completeness
        consistency_score = self._calculate_consistency_score(invoice_data)
        completeness = self._calculate_completeness_score(invoice_data)
        base_score = (consistency_score + completeness) / 2

        # Get OCR confidence if available
        ocr_confidence = invoice_data.get("processing_metadata", {}).get("ocr_confidence")

        if ocr_confidence is not None:
            ocr_confidence = float(ocr_confidence)
            # Use the higher of: base score or a weighted average that includes OCR confidence
            # This ensures OCR confidence can only increase or maintain the score
            weighted_with_ocr = (ocr_confidence * 0.5) + (base_score * 0.5)
            return max(base_score, weighted_with_ocr)

        return base_score

    def _calculate_consistency_score(self, invoice_data: Dict[str, Any]) -> float:
        """Calculate internal data consistency score (0-1)"""
        score = 1.0
        max_penalty = 0.0
        penalties = []

        # Helper function to add penalties
        def add_penalty(penalty: float, reason: str = ""):
            nonlocal max_penalty
            penalty = min(max(penalty, 0.0), 1.0)  # Ensure penalty is between 0 and 1
            penalties.append((penalty, reason))
            max_penalty = max(max_penalty, penalty)

        # 1. Check financial consistency (totals add up correctly)
        try:
            total_excl_vat = self._safe_decimal_conversion(invoice_data.get("total_excl_vat", 0))
            total_vat = self._safe_decimal_conversion(invoice_data.get("total_vat", 0))
            total_incl_vat = self._safe_decimal_conversion(invoice_data.get("total_incl_vat", 0))
            payable_amount = self._safe_decimal_conversion(invoice_data.get("payable_amount", 0))

            if total_incl_vat > 0:
                # Check if total_incl_vat = total_excl_vat + total_vat
                calculated_total = total_excl_vat + total_vat
                if calculated_total != 0:
                    difference = abs(calculated_total - total_incl_vat) / max(
                        abs(calculated_total), abs(total_incl_vat)
                    )
                    penalty = min(difference * 2, 0.3)  # Max penalty 0.3
                    if penalty > 0.01:  # Only add penalty if significant
                        add_penalty(
                            penalty,
                            f"Total incl. VAT ({total_incl_vat}) doesn't match sum of net + VAT ({calculated_total})",
                        )

                # Check if payable_amount matches total_incl_vat (if provided)
                if payable_amount > 0 and abs(payable_amount - total_incl_vat) > 0.01:
                    penalty = 0.1  # Small penalty for mismatch
                    add_penalty(
                        penalty, f"Payable amount ({payable_amount}) doesn't match total incl. VAT ({total_incl_vat})"
                    )
        except Exception as e:
            add_penalty(0.2, f"Error validating financial consistency: {str(e)}")

        # 2. Check line items consistency
        line_items = invoice_data.get("invoice_lines", [])
        if line_items:
            try:
                # Calculate sum of line totals
                line_totals_sum = sum(self._safe_decimal_conversion(item.get("line_total", 0)) for item in line_items)

                # Compare with total_excl_vat if available
                if total_excl_vat > 0 and line_totals_sum > 0:
                    difference = abs(line_totals_sum - total_excl_vat) / max(line_totals_sum, total_excl_vat)
                    if difference > 0.01:  # 1% tolerance
                        penalty = min(difference * 0.5, 0.2)  # Max penalty 0.2
                        add_penalty(
                            penalty,
                            f"Sum of line items ({line_totals_sum}) doesn't match total excl. VAT ({total_excl_vat})",
                        )

                # Check individual line calculations (quantity * unit_price = line_total)
                for i, item in enumerate(line_items):
                    try:
                        qty = self._safe_decimal_conversion(item.get("quantity", 0))
                        unit_price = self._safe_decimal_conversion(item.get("unit_price", 0))
                        line_total = self._safe_decimal_conversion(item.get("line_total", 0))

                        if qty > 0 and unit_price > 0 and line_total > 0:
                            calculated = qty * unit_price
                            if abs(calculated - line_total) > 0.01:  # Allow small floating point differences
                                penalty = 0.05  # Small penalty per line
                                add_penalty(
                                    penalty,
                                    f"Line {i+1}: Calculated total ({calculated}) doesn't match line_total ({line_total})",
                                )
                    except Exception as e:
                        add_penalty(0.05, f"Error validating line item {i+1}: {str(e)}")

            except Exception as e:
                add_penalty(0.1, f"Error calculating line items consistency: {str(e)}")

        # 3. Check VAT calculations if VAT rate is provided
        try:
            vat_breakdown = invoice_data.get("tax_breakdown", [])
            if vat_breakdown:
                for tax in vat_breakdown:
                    taxable = self._safe_decimal_conversion(tax.get("taxable_amount", 0))
                    tax_amount = self._safe_decimal_conversion(tax.get("tax_amount", 0))
                    rate = self._safe_decimal_conversion(tax.get("tax_rate", 0))

                    if taxable > 0 and rate > 0:
                        calculated_tax = taxable * rate
                        if abs(calculated_tax - tax_amount) > 0.01:  # Allow small differences
                            penalty = 0.1
                            add_penalty(
                                penalty,
                                f"VAT calculation error: {taxable} * {rate} should be {calculated_tax}, got {tax_amount}",
                            )
        except Exception as e:
            add_penalty(0.1, f"Error validating VAT calculations: {str(e)}")

        # 4. Check currency consistency
        try:
            currency = invoice_data.get("currency_code")
            supplier_country = self._get_nested_value(invoice_data, "supplier.country_code")

            if supplier_country and supplier_country in self.country_currencies:
                expected_currency = self.country_currencies[supplier_country]
                if currency != expected_currency:
                    penalty = 0.1
                    add_penalty(
                        penalty,
                        f"Currency {currency} doesn't match supplier country {supplier_country} (expected {expected_currency})",
                    )
        except Exception as e:
            add_penalty(0.05, f"Error validating currency consistency: {str(e)}")

        # Calculate final score (1 - total_penalty, but never below 0)
        total_penalty = min(sum(p[0] for p in penalties), 0.99)  # Cap at 0.99 to avoid returning 0
        final_score = max(1.0 - total_penalty, 0.01)  # Ensure we never return 0

        # Log warnings for any penalties applied (in debug mode)
        if penalties and logging.getLogger().isEnabledFor(logging.DEBUG):
            for penalty, reason in penalties:
                logging.debug(f"Consistency penalty ({penalty:.2f}): {reason}")

        return final_score

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
        if value is None:
            return Decimal("0")

        if isinstance(value, Decimal):
            return value

        if isinstance(value, (int, float)):
            return Decimal(str(value))

        if not isinstance(value, str):
            try:
                value = str(value)
            except:
                return Decimal("0")

        # Clean string value - keep digits, decimal points, commas, and minus sign
        cleaned = re.sub(r"[^\d.,\-]", "", value)
        if not cleaned:
            return Decimal("0")

        # Handle negative numbers
        is_negative = cleaned.startswith("-")
        if is_negative:
            cleaned = cleaned[1:]

        # Check for thousands separators vs decimal separators
        has_comma = "," in cleaned
        has_dot = "." in cleaned

        if has_comma and has_dot:
            # Determine which is the thousands separator and which is decimal
            if cleaned.find(",") > cleaned.find("."):
                # European format: 1.234,56
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                # US/UK format: 1,234.56
                cleaned = cleaned.replace(",", "")
        elif has_comma:
            # Could be either European decimal or thousands separator
            parts = cleaned.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Likely European decimal: 1234,56
                cleaned = cleaned.replace(",", ".")
            else:
                # Likely thousands separator: 1,234
                cleaned = cleaned.replace(",", "")
        # If only dot, it's treated as decimal

        # Restore negative sign if needed
        if is_negative:
            cleaned = f"-{cleaned}"

        try:
            return Decimal(cleaned)
        except (InvalidOperation, TypeError):
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
