"""
Pydantic models for European invoice data (EN 16931 compliant)
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
from datetime import datetime
import re


class InvoiceItem(BaseModel):
    """Invoice line item according to EN 16931"""

    line_id: Optional[str] = Field(None, description="Line identifier (BT-126)")
    description: str = Field(..., description="Item description (BT-153)")
    quantity: Optional[float] = Field(None, description="Invoiced quantity (BT-129)", ge=0)
    unit_code: Optional[str] = Field(None, description="Unit of measure code")
    unit_price: Optional[float] = Field(None, description="Unit price excluding VAT (BT-146)", ge=0)
    line_total: Optional[float] = Field(None, description="Line net amount (BT-131)", ge=0)
    vat_category: Optional[str] = Field(None, description="VAT category code (BT-151)")
    vat_rate: Optional[float] = Field(None, description="VAT rate (BT-152)", ge=0, le=100)

    @validator('unit_code')
    def validate_unit_code(cls, v):
        """Validate unit codes according to UN/ECE Recommendation 20"""
        if v and len(v) > 3:
            return v[:3]  # Truncate to 3 characters
        return v

    @validator('vat_category')
    def validate_vat_category(cls, v):
        """Validate VAT category codes"""
        valid_categories = ['S', 'Z', 'E', 'AE', 'K', 'G', 'O', 'L', 'M']
        if v and v not in valid_categories:
            return 'S'  # Default to standard rate
        return v


class TaxBreakdown(BaseModel):
    """Tax breakdown information"""

    taxable_amount: float = Field(..., description="Tax base amount", ge=0)
    tax_amount: float = Field(..., description="Tax amount", ge=0)
    tax_rate: float = Field(..., description="Tax rate percentage", ge=0, le=100)
    tax_category: str = Field(..., description="Tax category code")

    @validator('tax_category')
    def validate_tax_category(cls, v):
        """Validate tax category"""
        valid_categories = ['S', 'Z', 'E', 'AE', 'K', 'G', 'O', 'L', 'M']
        if v not in valid_categories:
            return 'S'
        return v


class Address(BaseModel):
    """Address information"""

    address_line_1: Optional[str] = Field(None, description="Address line 1")
    address_line_2: Optional[str] = Field(None, description="Address line 2")
    city: Optional[str] = Field(None, description="City")
    postal_code: Optional[str] = Field(None, description="Postal code")
    country_code: str = Field(..., description="Country code (ISO 3166-1 alpha-2)")

    @validator('country_code')
    def validate_country_code(cls, v):
        """Validate ISO country code"""
        if v and len(v) != 2:
            # Try to extract 2-letter code or default to common EU codes
            eu_codes = ['DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'PL', 'SE', 'DK', 'FI', 'EE', 'LV', 'LT']
            for code in eu_codes:
                if code.lower() in v.lower():
                    return code
            return 'DE'  # Default fallback
        return v.upper() if v else 'DE'


class Party(BaseModel):
    """Party information (supplier/customer)"""

    name: str = Field(..., description="Party name")
    vat_id: Optional[str] = Field(None, description="VAT identification number")
    tax_id: Optional[str] = Field(None, description="Tax identification number")
    commercial_id: Optional[str] = Field(None, description="Commercial registration number")
    address: Optional[Address] = Field(None, description="Party address")
    contact_email: Optional[str] = Field(None, description="Contact email")
    contact_phone: Optional[str] = Field(None, description="Contact phone")

    @validator('vat_id')
    def validate_vat_id(cls, v):
        """Validate VAT ID format"""
        if v:
            # Remove spaces and convert to uppercase
            v = re.sub(r'\s+', '', v.upper())
            # Basic VAT ID pattern validation
            if re.match(r'^[A-Z]{2}[A-Z0-9]+$', v):
                return v
        return v

    @validator('contact_email')
    def validate_email(cls, v):
        """Basic email validation"""
        if v and '@' not in v:
            return None
        return v


class PaymentInfo(BaseModel):
    """Payment information"""

    payment_terms: Optional[str] = Field(None, description="Payment terms")
    payment_due_date: Optional[str] = Field(None, description="Payment due date")
    payment_method: Optional[str] = Field(None, description="Payment method")
    bank_account: Optional[str] = Field(None, description="Bank account number")
    bank_code: Optional[str] = Field(None, description="Bank code")
    iban: Optional[str] = Field(None, description="IBAN")
    bic: Optional[str] = Field(None, description="BIC/SWIFT code")

    @validator('iban')
    def validate_iban(cls, v):
        """Basic IBAN validation"""
        if v:
            v = re.sub(r'\s+', '', v.upper())
            if len(v) >= 15 and len(v) <= 34 and v[:2].isalpha():
                return v
        return v


class InvoiceData(BaseModel):
    """Complete invoice data structure according to EN 16931"""

    # Core invoice information (BG-2)
    invoice_id: str = Field(..., description="Invoice number (BT-1)")
    issue_date: str = Field(..., description="Invoice issue date (BT-2)")
    invoice_type_code: Optional[str] = Field("380", description="Invoice type code (BT-3)")
    currency_code: str = Field("EUR", description="Invoice currency code (BT-5)")

    # Document references
    purchase_order_ref: Optional[str] = Field(None, description="Purchase order reference")
    contract_ref: Optional[str] = Field(None, description="Contract reference")

    # Parties (BG-4, BG-7)
    supplier: Party = Field(..., description="Supplier information")
    customer: Party = Field(..., description="Customer information")

    # Invoice lines (BG-25)
    invoice_lines: List[InvoiceItem] = Field(default_factory=list, description="Invoice line items")

    # Tax information (BG-23)
    tax_breakdown: List[TaxBreakdown] = Field(default_factory=list, description="VAT breakdown")

    # Totals (BG-22)
    total_excl_vat: Optional[float] = Field(None, description="Invoice total excluding VAT (BT-109)", ge=0)
    total_vat: Optional[float] = Field(None, description="Invoice total VAT amount (BT-110)", ge=0)
    total_incl_vat: float = Field(..., description="Invoice total including VAT (BT-112)", ge=0)
    payable_amount: Optional[float] = Field(None, description="Amount due for payment (BT-115)", ge=0)

    # Payment information
    payment_info: Optional[PaymentInfo] = Field(None, description="Payment information")

    # Additional fields
    notes: Optional[str] = Field(None, description="Invoice notes")
    language_code: Optional[str] = Field(None, description="Document language")

    # Processing metadata (not part of EN 16931)
    processing_metadata: Optional[dict] = Field(None, description="Processing metadata")
    confidence_score: Optional[float] = Field(None, description="Extraction confidence", ge=0, le=1)

    @validator('issue_date')
    def validate_date_format(cls, v):
        """Validate and normalize date format"""
        if not v:
            return v

        # Try to parse various date formats
        date_patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\d{1,2})[./-](\d{1,2})[./-](\d{4})',  # DD.MM.YYYY or DD/MM/YYYY
            r'(\d{1,2})[./-](\d{1,2})[./-](\d{2})',  # DD.MM.YY
        ]

        for pattern in date_patterns:
            match = re.search(pattern, v)
            if match:
                if len(match.groups()) == 3:
                    if len(match.group(1)) == 4:  # YYYY-MM-DD format
                        year, month, day = match.groups()
                    else:  # DD.MM.YYYY format
                        day, month, year = match.groups()
                        if len(year) == 2:
                            year = '20' + year if int(year) < 50 else '19' + year

                    try:
                        # Validate date
                        datetime(int(year), int(month), int(day))
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    except ValueError:
                        pass

        return v  # Return original if no valid format found

    @validator('currency_code')
    def validate_currency(cls, v):
        """Validate currency code"""
        common_currencies = ['EUR', 'USD', 'GBP', 'CHF', 'SEK', 'DKK', 'NOK', 'PLN', 'CZK']
        if v and v.upper() in common_currencies:
            return v.upper()
        return 'EUR'  # Default to EUR for European invoices

    @validator('invoice_type_code')
    def validate_invoice_type(cls, v):
        """Validate invoice type code according to UNTDID 1001"""
        valid_types = {
            '380': 'Commercial invoice',
            '381': 'Credit note',
            '383': 'Debit note',
            '384': 'Corrected invoice',
            '389': 'Self-billed invoice'
        }
        if v not in valid_types:
            return '380'  # Default to commercial invoice
        return v

    @validator('payable_amount')
    def set_payable_amount(cls, v, values):
        """Set payable amount to total if not provided"""
        if v is None and 'total_incl_vat' in values:
            return values['total_incl_vat']
        return v

    def calculate_totals(self):
        """Calculate totals from line items"""
        if self.invoice_lines:
            self.total_excl_vat = sum(
                (item.line_total or (item.quantity or 1) * (item.unit_price or 0))
                for item in self.invoice_lines
            )

            # Calculate VAT if tax breakdown is available
            if self.tax_breakdown:
                self.total_vat = sum(tax.tax_amount for tax in self.tax_breakdown)
                self.total_incl_vat = self.total_excl_vat + self.total_vat

            if not self.payable_amount:
                self.payable_amount = self.total_incl_vat

    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "allow"  # Allow extra fields for flexibility
        schema_extra = {
            "example": {
                "invoice_id": "INV-2024-001",
                "issue_date": "2024-01-15",
                "currency_code": "EUR",
                "supplier": {
                    "name": "Musterfirma GmbH",
                    "vat_id": "DE123456789",
                    "address": {
                        "address_line_1": "Musterstraße 1",
                        "city": "Berlin",
                        "postal_code": "10115",
                        "country_code": "DE"
                    }
                },
                "customer": {
                    "name": "Kunde AG",
                    "vat_id": "DE987654321"
                },
                "invoice_lines": [
                    {
                        "description": "Consulting Services",
                        "quantity": 10,
                        "unit_price": 100.0,
                        "line_total": 1000.0,
                        "vat_rate": 19.0
                    }
                ],
                "total_excl_vat": 1000.0,
                "total_vat": 190.0,
                "total_incl_vat": 1190.0
            }
        }