"""
Pydantic models for invoice data validation (EN 16931 compliant)
"""
from typing import List, Optional, Dict, Any
from datetime import date
from decimal import Decimal
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class CurrencyCode(str, Enum):
    """Supported currency codes"""
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    PLN = "PLN"
    SEK = "SEK"
    DKK = "DKK"


class CountryCode(str, Enum):
    """European country codes"""
    DE = "DE"  # Germany
    EE = "EE"  # Estonia
    EN = "GB"  # United Kingdom
    FR = "FR"  # France
    IT = "IT"  # Italy
    ES = "ES"  # Spain
    NL = "NL"  # Netherlands
    AT = "AT"  # Austria
    BE = "BE"  # Belgium
    PL = "PL"  # Poland
    FI = "FI"  # Finland
    SE = "SE"  # Sweden
    DK = "DK"  # Denmark


class InvoiceTypeCode(str, Enum):
    """Invoice type codes according to EN 16931"""
    COMMERCIAL_INVOICE = "380"
    CREDIT_NOTE = "381"
    DEBIT_NOTE = "383"
    CORRECTED_INVOICE = "384"


class VATCategory(str, Enum):
    """VAT category codes"""
    STANDARD_RATE = "S"
    ZERO_RATED = "Z"
    EXEMPT = "E"
    REVERSE_CHARGE = "AE"
    NOT_SUBJECT = "O"


class Party(BaseModel):
    """Party information (supplier/customer)"""
    name: str = Field(..., description="Party name", max_length=200)
    vat_id: Optional[str] = Field(None, description="VAT identification number")
    tax_id: Optional[str] = Field(None, description="Tax identification number")
    address_line: Optional[str] = Field(None, description="Address line", max_length=300)
    city: Optional[str] = Field(None, description="City", max_length=100)
    postal_code: Optional[str] = Field(None, description="Postal code", max_length=20)
    country_code: CountryCode = Field(..., description="Country code ISO 3166-1")
    
    @validator('vat_id')
    def validate_vat_id(cls, v):
        """Validate VAT ID format"""
        if v and len(v) > 2:
            # Basic VAT validation - starts with country code
            if not v[:2].isalpha():
                raise ValueError('VAT ID must start with country code')
        return v


class InvoiceItem(BaseModel):
    """Invoice line item (BG-25)"""
    line_id: str = Field(..., description="Line identifier (BT-126)")
    description: str = Field(..., description="Item name (BT-153)", max_length=500)
    quantity: Decimal = Field(..., description="Invoiced quantity (BT-129)", ge=0)
    unit_code: str = Field(default="C62", description="Unit of measure code", max_length=10)
    unit_price: Decimal = Field(..., description="Item net price (BT-146)", ge=0)
    line_total: Decimal = Field(..., description="Line net amount (BT-131)", ge=0)
    vat_category: VATCategory = Field(default=VATCategory.STANDARD_RATE, description="VAT category (BT-151)")
    vat_rate: Decimal = Field(..., description="VAT rate (BT-152)", ge=0, le=1)
    
    @root_validator
    def validate_line_total(cls, values):
        """Validate that line total matches quantity * unit price"""
        quantity = values.get('quantity')
        unit_price = values.get('unit_price')
        line_total = values.get('line_total')
        
        if quantity is not None and unit_price is not None and line_total is not None:
            calculated_total = quantity * unit_price
            # Allow small rounding differences
            if abs(calculated_total - line_total) > Decimal('0.01'):
                values['line_total'] = calculated_total
        
        return values


class TaxBreakdown(BaseModel):
    """Tax breakdown (BG-23)"""
    taxable_amount: Decimal = Field(..., description="VAT category taxable amount (BT-116)", ge=0)
    tax_amount: Decimal = Field(..., description="VAT category tax amount (BT-117)", ge=0)
    tax_rate: Decimal = Field(..., description="VAT category rate (BT-119)", ge=0, le=1)
    tax_category: VATCategory = Field(..., description="VAT category code (BT-118)")
    
    @root_validator
    def validate_tax_calculation(cls, values):
        """Validate tax calculation"""
        taxable_amount = values.get('taxable_amount')
        tax_rate = values.get('tax_rate')
        tax_amount = values.get('tax_amount')
        
        if all(v is not None for v in [taxable_amount, tax_rate, tax_amount]):
            calculated_tax = taxable_amount * tax_rate
            # Allow small rounding differences
            if abs(calculated_tax - tax_amount) > Decimal('0.01'):
                values['tax_amount'] = calculated_tax
        
        return values


class PaymentTerms(BaseModel):
    """Payment terms information"""
    payment_due_date: Optional[str] = Field(None, description="Payment due date (BT-9)")
    payment_terms: Optional[str] = Field(None, description="Payment terms (BT-20)", max_length=500)
    bank_account: Optional[str] = Field(None, description="Bank account number")
    bank_name: Optional[str] = Field(None, description="Bank name")
    swift_code: Optional[str] = Field(None, description="SWIFT/BIC code")


class InvoiceData(BaseModel):
    """
    Complete invoice data model compliant with EN 16931
    European standard for electronic invoicing
    """
    
    # Invoice identification (BG-2)
    invoice_id: str = Field(..., description="Invoice number (BT-1)", max_length=50)
    issue_date: str = Field(..., description="Invoice issue date (BT-2)")
    invoice_type_code: InvoiceTypeCode = Field(
        default=InvoiceTypeCode.COMMERCIAL_INVOICE,
        description="Invoice type code (BT-3)"
    )
    currency_code: CurrencyCode = Field(default=CurrencyCode.EUR, description="Invoice currency code (BT-5)")
    
    # Document references
    purchase_order_reference: Optional[str] = Field(None, description="Buyer reference (BT-10)", max_length=50)
    contract_reference: Optional[str] = Field(None, description="Contract reference (BT-12)", max_length=50)
    
    # Parties
    supplier: Party = Field(..., description="Seller (BG-4)")
    customer: Party = Field(..., description="Buyer (BG-7)")
    
    # Invoice lines
    invoice_lines: List[InvoiceItem] = Field(default_factory=list, description="Invoice lines (BG-25)")
    
    # Document totals (BG-22)
    total_excl_vat: Decimal = Field(..., description="Sum of line net amounts (BT-106)", ge=0)
    total_vat: Decimal = Field(..., description="Total VAT amount (BT-110)", ge=0)
    total_incl_vat: Decimal = Field(..., description="Invoice total with VAT (BT-112)", ge=0)
    payable_amount: Optional[Decimal] = Field(None, description="Amount due for payment (BT-115)", ge=0)
    
    # VAT breakdown
    tax_breakdown: List[TaxBreakdown] = Field(default_factory=list, description="VAT breakdown (BG-23)")
    
    # Payment information
    payment_terms: Optional[PaymentTerms] = Field(None, description="Payment terms")
    
    # Additional fields for processing
    detected_language: Optional[str] = Field(None, description="Detected language")
    processing_metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")
    
    @validator('issue_date')
    def validate_issue_date(cls, v):
        """Validate date format"""
        from datetime import datetime
        try:
            # Accept various date formats
            for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%d-%m-%Y']:
                try:
                    parsed_date = datetime.strptime(v, fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            raise ValueError('Invalid date format')
        except Exception:
            raise ValueError('Invalid date format, expected YYYY-MM-DD or DD.MM.YYYY or DD/MM/YYYY')
    
    @root_validator
    def validate_totals(cls, values):
        """Validate invoice totals consistency"""
        total_excl_vat = values.get('total_excl_vat')
        total_vat = values.get('total_vat')
        total_incl_vat = values.get('total_incl_vat')
        
        if all(v is not None for v in [total_excl_vat, total_vat, total_incl_vat]):
            calculated_total = total_excl_vat + total_vat
            # Allow small rounding differences
            if abs(calculated_total - total_incl_vat) > Decimal('0.01'):
                values['total_incl_vat'] = calculated_total
        
        # Set payable amount if not provided
        if values.get('payable_amount') is None:
            values['payable_amount'] = values.get('total_incl_vat')
        
        return values
    
    @root_validator
    def validate_line_items_total(cls, values):
        """Validate that line items sum matches document total"""
        invoice_lines = values.get('invoice_lines', [])
        total_excl_vat = values.get('total_excl_vat')
        
        if invoice_lines and total_excl_vat is not None:
            calculated_total = sum(item.line_total for item in invoice_lines)
            # Allow small rounding differences
            if abs(calculated_total - total_excl_vat) > Decimal('0.01'):
                values['total_excl_vat'] = calculated_total
                
                # Recalculate VAT and total
                total_vat = values.get('total_vat', Decimal('0'))
                values['total_incl_vat'] = calculated_total + total_vat
                values['payable_amount'] = values['total_incl_vat']
        
        return values
    
    def to_en16931_xml(self) -> str:
        """Convert to EN 16931 compliant XML format"""
        # This would implement full EN 16931 XML serialization
        # For now, return a basic structure
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Invoice xmlns="urn:oasis:names:specification:ubl:schema:xsd:Invoice-2">
    <ID>{self.invoice_id}</ID>
    <IssueDate>{self.issue_date}</IssueDate>
    <InvoiceTypeCode>{self.invoice_type_code.value}</InvoiceTypeCode>
    <DocumentCurrencyCode>{self.currency_code.value}</DocumentCurrencyCode>
    <!-- Additional EN 16931 elements would be added here -->
</Invoice>"""
    
    def to_peppol_bis(self) -> Dict[str, Any]:
        """Convert to PEPPOL BIS 3.0 format"""
        return {
            "invoice_id": self.invoice_id,
            "issue_date": self.issue_date,
            "currency": self.currency_code.value,
            "supplier": self.supplier.dict(),
            "customer": self.customer.dict(),
            "lines": [line.dict() for line in self.invoice_lines],
            "totals": {
                "net": float(self.total_excl_vat),
                "vat": float(self.total_vat),
                "gross": float(self.total_incl_vat)
            }
        }
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            Decimal: lambda v: float(v)
        }
        schema_extra = {
            "example": {
                "invoice_id": "INV-2024-001",
                "issue_date": "2024-01-15",
                "currency_code": "EUR",
                "supplier": {
                    "name": "Example Company GmbH",
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
                        "description": "Professional Services",
                        "quantity": "1.00",
                        "unit_price": "100.00",
                        "line_total": "100.00",
                        "vat_rate": "0.20"
                    }
                ],
                "total_excl_vat": "100.00",
                "total_vat": "20.00",
                "total_incl_vat": "120.00"
            }
        }


class InvoiceBatch(BaseModel):
    """Model for batch processing results"""
    batch_id: str = Field(..., description="Batch identifier")
    total_files: int = Field(..., description="Total files in batch")
    processed_files: int = Field(..., description="Successfully processed files")
    failed_files: int = Field(..., description="Failed files")
    processing_time: float = Field(..., description="Total processing time in seconds")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Individual file results")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_files == 0:
            return 0.0
        return self.processed_files / self.total_files


class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field name with error")
    message: str = Field(..., description="Error message")
    value: Optional[Any] = Field(None, description="Invalid value")


class InvoiceValidationResult(BaseModel):
    """Invoice validation result"""
    is_valid: bool = Field(..., description="Whether invoice is valid")
    errors: List[ValidationError] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    completeness_score: float = Field(..., description="Data completeness score 0-1")
    confidence_score: float = Field(..., description="Extraction confidence score 0-1")
