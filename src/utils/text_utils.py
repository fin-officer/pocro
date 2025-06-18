"""
Text processing utilities
"""
import re
from typing import List, Optional
import unicodedata


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text"""
    # Pattern to match numbers with optional decimal places
    pattern = r'-?\d+(?:[.,]\d+)?'
    matches = re.findall(pattern, text)

    numbers = []
    for match in matches:
        try:
            # Replace comma with dot for decimal
            normalized = match.replace(',', '.')
            numbers.append(float(normalized))
        except ValueError:
            continue

    return numbers


def extract_dates(text: str) -> List[str]:
    """Extract dates from text"""
    date_patterns = [
        r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',  # DD.MM.YYYY, DD/MM/YYYY
        r'\d{4}[./-]\d{1,2}[./-]\d{1,2}',    # YYYY.MM.DD, YYYY/MM/DD
        r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}' # DD Month YYYY
    ]

    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)

    return dates


def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


def extract_phone_numbers(text: str) -> List[str]:
    """Extract phone numbers from text"""
    phone_patterns = [
        r'\+\d{1,3}[\s.-]?\d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{1,9}',  # International
        r'\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,6}',                  # National
    ]

    phones = []
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        phones.extend(matches)

    return phones


def detect_invoice_type(text: str) -> str:
    """Detect invoice type from text content"""
    text_lower = text.lower()

    if any(word in text_lower for word in ['credit', 'kredit', 'gutschrift', 'kreedit']):
        return 'credit_note'
    elif any(word in text_lower for word in ['debit', 'lastschrift', 'deebet']):
        return 'debit_note'
    elif any(word in text_lower for word in ['proforma', 'pro forma']):
        return 'proforma'
    else:
        return 'invoice'


def normalize_vat_number(vat_number: str) -> Optional[str]:
    """Normalize VAT number format"""
    if not vat_number:
        return None

    # Remove spaces and convert to uppercase
    normalized = re.sub(r'\s+', '', vat_number.upper())

    # Check if it starts with country code
    if re.match(r'^[A-Z]{2}[A-Z0-9]+, normalized):
        return normalized

    return vat_number  # Return original if not standard format