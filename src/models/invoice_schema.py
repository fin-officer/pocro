"""
Pydantic models for invoice data validation and serialization.
"""
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class PaymentMethod(str, Enum):
    """Supported payment methods"""
    BANK_TRANSFER = "bank_transfer"
    CREDIT_CARD = "credit_card"
    PAYPAL = "paypal"
    OTHER = "other"


class Currency(str, Enum):
    """Supported currencies"""
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    CHF = "CHF"


class Address(BaseModel):
    """Address information"""
    street: str
    city: str
    postal_code: str
    country: str
    state: Optional[str] = None


class LineItem(BaseModel):
    """Individual line item in an invoice"""
    description: str
    quantity: float = 1.0
    unit_price: float = Field(..., gt=0, description="Price per unit")
    tax_rate: float = Field(0.0, ge=0, le=1, description="Tax rate as a decimal (e.g., 0.19 for 19%)")
    amount: float = Field(..., gt=0, description="Total amount including tax")


class InvoiceData(BaseModel):
    """Main invoice data model"""
    # Basic information
    invoice_number: str
    issue_date: date
    due_date: date
    currency: Currency = Currency.EUR
    
    # Parties
    supplier: Dict[str, str]  # Name, address, tax_id, etc.
    customer: Dict[str, str]  # Name, address, tax_id, etc.
    
    # Financials
    subtotal: float = Field(..., gt=0)
    tax_amount: float = Field(0.0, ge=0)
    total_amount: float = Field(..., gt=0)
    amount_paid: float = Field(0.0, ge=0)
    balance_due: float = Field(0.0, ge=0)
    
    # Items
    line_items: List[LineItem] = []
    
    # Payment information
    payment_terms: Optional[str] = None
    payment_method: Optional[PaymentMethod] = None
    payment_reference: Optional[str] = None
    
    # Metadata
    notes: Optional[str] = None
    reference: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }
    
    @validator('due_date')
    def due_date_after_issue_date(cls, v, values):
        if 'issue_date' in values and v < values['issue_date']:
            raise ValueError('due_date must be after issue_date')
        return v
    
    @validator('total_amount')
    def validate_totals(cls, v, values):
        if 'subtotal' in values and 'tax_amount' in values:
            expected_total = values['subtotal'] + values['tax_amount']
            if abs(v - expected_total) > 0.01:  # Allow for floating point errors
                raise ValueError(f'Total amount {v} does not match subtotal + tax: {expected_total}')
        return v

