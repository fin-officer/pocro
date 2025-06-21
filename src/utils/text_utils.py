"""
Text processing utilities for invoice data extraction
"""

import logging
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class TextProcessor:
    """Advanced text processing utilities for invoice data"""

    def __init__(self):
        # Currency symbols and codes
        self.currency_symbols = {
            "€": "EUR",
            "$": "USD",
            "£": "GBP",
            "¥": "JPY",
            "₹": "INR",
            "₽": "RUB",
            "₩": "KRW",
            "₴": "UAH",
        }

        # Common invoice number patterns
        self.invoice_patterns = [
            r"(?:Invoice|Rechnung|Arve|Factura|Fattura)[\s#:]*([A-Z0-9\-/\.]+)",
            r"(?:No|Nr|Number|Numero)[\s.:]*([A-Z0-9\-/\.]+)",
            r"([A-Z]{1,4}\-\d{4,})",  # ABC-1234
            r"(\d{4,})",  # 1234567
            r"([A-Z]{1,4}\d{4,})",  # ABC1234
            r"(\d{4}/\d{4})",  # 2024/0001
            r"([A-Z]{1,4}/\d{4,})",  # INV/2024001
        ]

        # Date patterns for different locales
        self.date_patterns = [
            (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),  # 2024-01-15
            (r"(\d{2})\.(\d{2})\.(\d{4})", "%d.%m.%Y"),  # 15.01.2024
            (r"(\d{2})/(\d{2})/(\d{4})", "%d/%m/%Y"),  # 15/01/2024
            (r"(\d{2})-(\d{2})-(\d{4})", "%d-%m-%Y"),  # 15-01-2024
            (r"(\d{1,2})\s+(\w+)\s+(\d{4})", "%d %B %Y"),  # 15 January 2024
            (r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", "%B %d %Y"),  # January 15, 2024
        ]

        # Amount patterns
        self.amount_patterns = [
            r"([€$£¥₹₽₩₴])\s*([0-9\s.,]+)",  # Symbol before
            r"([0-9\s.,]+)\s*([€$£¥₹₽₩₴])",  # Symbol after
            r"([0-9\s.,]+)\s*(EUR|USD|GBP|JPY|PLN|SEK|DKK|NOK)",  # Currency code
            r"([0-9]{1,3}(?:[.,]\d{3})*[.,]\d{2})",  # Standard amount format
        ]

        # VAT ID patterns by country
        self.vat_patterns = {
            "DE": r"DE[0-9]{9}",  # Germany
            "EE": r"EE[0-9]{9}",  # Estonia
            "GB": r"GB[0-9]{9}",  # United Kingdom
            "FR": r"FR[0-9A-Z]{2}[0-9]{9}",  # France
            "IT": r"IT[0-9]{11}",  # Italy
            "ES": r"ES[0-9A-Z][0-9]{7}[0-9A-Z]",  # Spain
            "NL": r"NL[0-9]{9}B[0-9]{2}",  # Netherlands
            "AT": r"ATU[0-9]{8}",  # Austria
            "BE": r"BE[0-9]{10}",  # Belgium
            "PL": r"PL[0-9]{10}",  # Poland
            "FI": r"FI[0-9]{8}",  # Finland
            "SE": r"SE[0-9]{12}",  # Sweden
            "DK": r"DK[0-9]{8}",  # Denmark
        }

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        try:
            # Normalize unicode characters
            text = unicodedata.normalize("NFKC", text)

            # Remove excessive whitespace
            text = re.sub(r"\s+", " ", text)

            # Strip leading/trailing whitespace
            text = text.strip()

            return text

        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return text

    def extract_invoice_number(self, text: str) -> Optional[str]:
        """
        Extract invoice number from text

        Args:
            text: Input text

        Returns:
            Extracted invoice number or None
        """
        try:
            for pattern in self.invoice_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    invoice_number = match.group(1) if match.lastindex else match.group()
                    return self.clean_text(invoice_number)

            return None

        except Exception as e:
            logger.error(f"Invoice number extraction failed: {e}")
            return None

    def extract_dates(self, text: str) -> List[str]:
        """
        Extract all dates from text

        Args:
            text: Input text

        Returns:
            List of dates in ISO format (YYYY-MM-DD)
        """
        dates = []

        try:
            for pattern, date_format in self.date_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    try:
                        if date_format == "%d %B %Y":
                            # Handle month names
                            day, month_name, year = match.groups()
                            date_str = f"{day} {month_name} {year}"
                        elif date_format == "%B %d %Y":
                            # Handle month names
                            month_name, day, year = match.groups()
                            date_str = f"{month_name} {day} {year}"
                        else:
                            # Standard numeric formats
                            date_str = match.group()

                        # Parse and format date
                        parsed_date = datetime.strptime(date_str, date_format)
                        iso_date = parsed_date.strftime("%Y-%m-%d")

                        if iso_date not in dates:
                            dates.append(iso_date)

                    except ValueError:
                        continue

            return sorted(dates)

        except Exception as e:
            logger.error(f"Date extraction failed: {e}")
            return []

    def extract_amounts(self, text: str) -> List[Dict[str, Union[str, float]]]:
        """
        Extract monetary amounts from text

        Args:
            text: Input text

        Returns:
            List of dictionaries with amount and currency
        """
        amounts = []

        try:
            for pattern in self.amount_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    try:
                        groups = match.groups()

                        if len(groups) == 2:
                            # Currency symbol and amount
                            if groups[0] in self.currency_symbols:
                                currency = self.currency_symbols[groups[0]]
                                amount_str = groups[1]
                            elif groups[1] in self.currency_symbols:
                                currency = self.currency_symbols[groups[1]]
                                amount_str = groups[0]
                            elif groups[1] in ["EUR", "USD", "GBP", "JPY", "PLN", "SEK", "DKK", "NOK"]:
                                currency = groups[1]
                                amount_str = groups[0]
                            else:
                                continue
                        else:
                            # Amount only, assume EUR for European invoices
                            currency = "EUR"
                            amount_str = groups[0]

                        # Parse amount
                        amount_value = self.parse_amount(amount_str)

                        if amount_value is not None and amount_value > 0:
                            amounts.append(
                                {"amount": float(amount_value), "currency": currency, "raw_text": match.group()}
                            )

                    except Exception:
                        continue

            # Remove duplicates and sort by amount
            unique_amounts = []
            seen = set()

            for item in amounts:
                key = (item["amount"], item["currency"])
                if key not in seen:
                    seen.add(key)
                    unique_amounts.append(item)

            return sorted(unique_amounts, key=lambda x: x["amount"], reverse=True)

        except Exception as e:
            logger.error(f"Amount extraction failed: {e}")
            return []

    def parse_amount(self, amount_str: str) -> Optional[Decimal]:
        """
        Parse amount string to Decimal

        Args:
            amount_str: Amount string

        Returns:
            Parsed amount as Decimal or None
        """
        try:
            if not amount_str:
                return None

            # Clean the string
            cleaned = re.sub(r"[^\d.,\-]", "", amount_str.strip())

            if not cleaned:
                return None

            # Handle different decimal separators
            if "," in cleaned and "." in cleaned:
                # European format: 1.234,56
                if cleaned.rfind(",") > cleaned.rfind("."):
                    cleaned = cleaned.replace(".", "").replace(",", ".")
                else:
                    # US format: 1,234.56
                    cleaned = cleaned.replace(",", "")
            elif "," in cleaned:
                # Check if comma is decimal separator
                parts = cleaned.split(",")
                if len(parts) == 2 and len(parts[1]) <= 2:
                    # Decimal separator
                    cleaned = cleaned.replace(",", ".")
                else:
                    # Thousands separator
                    cleaned = cleaned.replace(",", "")

            return Decimal(cleaned)

        except (InvalidOperation, ValueError) as e:
            logger.debug(f"Amount parsing failed for '{amount_str}': {e}")
            return None

    def extract_vat_ids(self, text: str) -> List[Dict[str, str]]:
        """
        Extract VAT IDs from text

        Args:
            text: Input text

        Returns:
            List of dictionaries with VAT ID and country
        """
        vat_ids = []

        try:
            for country, pattern in self.vat_patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    vat_id = match.group().upper()
                    vat_ids.append({"vat_id": vat_id, "country": country})

            return vat_ids

        except Exception as e:
            logger.error(f"VAT ID extraction failed: {e}")
            return []

    def detect_language(self, text: str) -> str:
        """
        Detect language of text using keyword analysis

        Args:
            text: Input text

        Returns:
            Language code (en, de, et, fr, etc.)
        """
        try:
            text_lower = text.lower()

            # Language keywords
            language_keywords = {
                "de": [
                    "rechnung",
                    "datum",
                    "betrag",
                    "mwst",
                    "ust",
                    "gesamt",
                    "summe",
                    "netto",
                    "brutto",
                    "lieferant",
                    "kunde",
                    "steuernummer",
                    "ustidnr",
                    "rechnungsnummer",
                    "zahlbar",
                    "fällig",
                ],
                "et": [
                    "arve",
                    "kuupäev",
                    "summa",
                    "käibemaks",
                    "kokku",
                    "maksukohustuslane",
                    "reg",
                    "müüja",
                    "ostja",
                    "maksetähtaeg",
                    "number",
                ],
                "en": [
                    "invoice",
                    "date",
                    "amount",
                    "vat",
                    "total",
                    "subtotal",
                    "tax",
                    "supplier",
                    "customer",
                    "due",
                    "payment",
                    "number",
                    "net",
                    "gross",
                ],
                "fr": [
                    "facture",
                    "date",
                    "montant",
                    "tva",
                    "total",
                    "fournisseur",
                    "client",
                    "échéance",
                    "paiement",
                    "numéro",
                    "net",
                    "brut",
                ],
                "es": [
                    "factura",
                    "fecha",
                    "importe",
                    "iva",
                    "total",
                    "proveedor",
                    "cliente",
                    "vencimiento",
                    "pago",
                    "número",
                    "neto",
                    "bruto",
                ],
                "it": [
                    "fattura",
                    "data",
                    "importo",
                    "iva",
                    "totale",
                    "fornitore",
                    "cliente",
                    "scadenza",
                    "pagamento",
                    "numero",
                    "netto",
                    "lordo",
                ],
                "nl": [
                    "factuur",
                    "datum",
                    "bedrag",
                    "btw",
                    "totaal",
                    "leverancier",
                    "klant",
                    "vervaldatum",
                    "betaling",
                    "nummer",
                    "netto",
                    "bruto",
                ],
            }

            # Count matches for each language
            scores = {}
            for lang, keywords in language_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                scores[lang] = score

            # Return language with highest score
            if scores:
                detected_lang = max(scores, key=scores.get)
                max_score = scores[detected_lang]

                # Only return if confidence is reasonable
                if max_score >= 2:
                    return detected_lang

            # Default to English
            return "en"

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"

    def extract_companies(self, text: str) -> List[Dict[str, str]]:
        """
        Extract company information from text

        Args:
            text: Input text

        Returns:
            List of company information dictionaries
        """
        companies = []

        try:
            # Company suffixes
            suffixes = [
                r"GmbH",
                r"AG",
                r"Ltd",
                r"Limited",
                r"Inc",
                r"Corp",
                r"Corporation",
                r"OÜ",
                r"AS",
                r"OY",
                r"AB",
                r"ApS",
                r"A/S",
                r"SA",
                r"SL",
                r"SRL",
                r"LLC",
                r"LLP",
                r"LP",
                r"PLC",
                r"BV",
                r"NV",
            ]

            suffix_pattern = "|".join(suffixes)

            # Pattern to match company names
            pattern = rf"([A-ZÀ-ÿ][a-zA-ZÀ-ÿ\s&\-\.]+\s+(?:{suffix_pattern})\.?)"

            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                company_name = self.clean_text(match.group(1))

                if len(company_name) > 3:  # Filter out very short matches
                    companies.append({"name": company_name, "position": match.start()})

            # Remove duplicates and sort by position
            unique_companies = []
            seen = set()

            for company in companies:
                if company["name"] not in seen:
                    seen.add(company["name"])
                    unique_companies.append(company)

            return sorted(unique_companies, key=lambda x: x["position"])

        except Exception as e:
            logger.error(f"Company extraction failed: {e}")
            return []

    def extract_addresses(self, text: str) -> List[str]:
        """
        Extract addresses from text

        Args:
            text: Input text

        Returns:
            List of potential addresses
        """
        addresses = []

        try:
            # Pattern for postal codes (European formats)
            postal_patterns = [
                r"\b\d{5}\b",  # Germany (12345)
                r"\b\d{5}\s+[A-Z]+\b",  # Estonia (12345 Tallinn)
                r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s+\d[A-Z]{2}\b",  # UK (SW1A 1AA)
                r"\b\d{2}-\d{3}\b",  # Poland (12-345)
                r"\b\d{3}\s?\d{2}\b",  # Sweden/Norway (123 45)
            ]

            for pattern in postal_patterns:
                matches = re.finditer(pattern, text)

                for match in matches:
                    # Extract surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 50)

                    context = text[start:end].strip()
                    lines = context.split("\n")

                    # Look for address-like patterns
                    for line in lines:
                        line = line.strip()
                        if len(line) > 10 and match.group() in line:
                            addresses.append(self.clean_text(line))
                            break

            return list(set(addresses))  # Remove duplicates

        except Exception as e:
            logger.error(f"Address extraction failed: {e}")
            return []

    def normalize_line_endings(self, text: str) -> str:
        """
        Normalize line endings in text

        Args:
            text: Input text

        Returns:
            Text with normalized line endings
        """
        # Replace various line ending types with \n
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\r", "\n", text)

        # Remove excessive line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def extract_emails(self, text: str) -> List[str]:
        """
        Extract email addresses from text

        Args:
            text: Input text

        Returns:
            List of email addresses
        """
        try:
            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            emails = re.findall(email_pattern, text)
            return list(set(emails))  # Remove duplicates

        except Exception as e:
            logger.error(f"Email extraction failed: {e}")
            return []

    def extract_phone_numbers(self, text: str) -> List[str]:
        """
        Extract phone numbers from text

        Args:
            text: Input text

        Returns:
            List of phone numbers
        """
        try:
            # European phone number patterns
            patterns = [
                r"\+\d{1,3}[-.\s]?\d{1,14}",  # International format
                r"\b\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b",  # Local format
                r"\(\d{2,4}\)\s?\d{3,8}",  # Format with parentheses
            ]

            phone_numbers = []

            for pattern in patterns:
                matches = re.findall(pattern, text)
                phone_numbers.extend(matches)

            # Clean and deduplicate
            cleaned_numbers = []
            for number in phone_numbers:
                cleaned = re.sub(r"[^\d+]", "", number)
                if len(cleaned) >= 7:  # Minimum phone number length
                    cleaned_numbers.append(number.strip())

            return list(set(cleaned_numbers))

        except Exception as e:
            logger.error(f"Phone number extraction failed: {e}")
            return []

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using simple word overlap

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        try:
            if not text1 or not text2:
                return 0.0

            # Normalize texts
            words1 = set(re.findall(r"\b\w+\b", text1.lower()))
            words2 = set(re.findall(r"\b\w+\b", text2.lower()))

            if not words1 or not words2:
                return 0.0

            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Text similarity calculation failed: {e}")
            return 0.0


# Global text processor instance
text_processor = TextProcessor()


def extract_key_invoice_data(text: str) -> Dict[str, Any]:
    """
    Extract key invoice data using text processing

    Args:
        text: Invoice text

    Returns:
        Dictionary with extracted data
    """
    try:
        return {
            "invoice_number": text_processor.extract_invoice_number(text),
            "dates": text_processor.extract_dates(text),
            "amounts": text_processor.extract_amounts(text),
            "vat_ids": text_processor.extract_vat_ids(text),
            "companies": text_processor.extract_companies(text),
            "language": text_processor.detect_language(text),
            "emails": text_processor.extract_emails(text),
            "phone_numbers": text_processor.extract_phone_numbers(text),
        }

    except Exception as e:
        logger.error(f"Key data extraction failed: {e}")
        return {}


def clean_ocr_text(text: str) -> str:
    """
    Clean OCR text for better processing

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text
    """
    try:
        # Basic cleaning
        cleaned = text_processor.clean_text(text)

        # Normalize line endings
        cleaned = text_processor.normalize_line_endings(cleaned)

        # Fix common OCR errors
        ocr_fixes = {
            r"\b0(?=[A-Z])\b": "O",  # 0 -> O before uppercase
            r"\b1(?=[a-z])\b": "l",  # 1 -> l before lowercase
            r"\bS(?=\d)\b": "5",  # S -> 5 before digits
            r"\bO(?=\d)\b": "0",  # O -> 0 before digits
        }

        for pattern, replacement in ocr_fixes.items():
            cleaned = re.sub(pattern, replacement, cleaned)

        return cleaned

    except Exception as e:
        logger.error(f"OCR text cleaning failed: {e}")
        return text
