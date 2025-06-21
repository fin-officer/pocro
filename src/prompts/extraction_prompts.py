"""
LLM prompts for invoice data extraction
"""

import json
from typing import Any, Dict

from src.models.invoice_schema import InvoiceData


def get_extraction_prompt(invoice_text: str, detected_language: str = "en") -> str:
    """
    Generate language-specific extraction prompt

    Args:
        invoice_text: Raw OCR text from invoice
        detected_language: Detected language code (en, de, et)

    Returns:
        Formatted prompt for LLM
    """
    # Get the JSON schema
    schema = InvoiceData.schema()
    schema_json = json.dumps(schema, indent=2)

    # Language-specific prompts
    prompts = {"en": ENGLISH_EXTRACTION_PROMPT, "de": GERMAN_EXTRACTION_PROMPT, "et": ESTONIAN_EXTRACTION_PROMPT}

    # Get prompt template for detected language, fallback to English
    prompt_template = prompts.get(detected_language, prompts["en"])

    return prompt_template.format(schema=schema_json, invoice_text=invoice_text)


ENGLISH_EXTRACTION_PROMPT = """You are an AI assistant specialized in extracting structured data from European invoices. Your task is to extract information from the provided invoice text and return it as valid JSON matching the provided schema.

CRITICAL INSTRUCTIONS:
- Return ONLY valid JSON, no additional text or explanations
- Use exact field names from the schema
- Convert all monetary amounts to numbers (remove currency symbols)
- Use YYYY-MM-DD format for all dates
- If information is not available, use null
- For decimal numbers, use dots as decimal separators
- Ensure all required fields are included

EUROPEAN INVOICE STANDARDS:
- Follow EN 16931 standard for electronic invoicing
- VAT rates are typically 0.19-0.25 for most EU countries
- Invoice numbers usually follow patterns like INV-2024-001 or similar
- Supplier and customer information should include VAT IDs when available

JSON SCHEMA:
{schema}

INVOICE TEXT TO EXTRACT:
{invoice_text}

JSON OUTPUT:"""


GERMAN_EXTRACTION_PROMPT = """Du bist ein KI-Assistent, der auf die Extraktion strukturierter Daten aus europäischen Rechnungen spezialisiert ist. Deine Aufgabe ist es, Informationen aus dem bereitgestellten Rechnungstext zu extrahieren und als gültiges JSON zurückzugeben, das dem bereitgestellten Schema entspricht.

WICHTIGE ANWEISUNGEN:
- Gib NUR gültiges JSON zurück, keinen zusätzlichen Text oder Erklärungen
- Verwende die exakten Feldnamen aus dem Schema
- Konvertiere alle Geldbeträge in Zahlen (entferne Währungssymbole)
- Verwende das Format YYYY-MM-DD für alle Daten
- Wenn Informationen nicht verfügbar sind, verwende null
- Für Dezimalzahlen verwende Punkte als Dezimaltrennzeichen
- Stelle sicher, dass alle erforderlichen Felder enthalten sind

DEUTSCHE RECHNUNGSKONVENTIONEN:
- MwSt./USt. = VAT (Value Added Tax)
- Datum = issue_date
- Rechnungsnummer = invoice_id
- Lieferant/Verkäufer = supplier
- Kunde/Käufer = customer
- Nettobetrag = total_excl_vat
- Bruttobetrag = total_incl_vat
- Steuerbetrag = total_vat
- Standardmäßiger MwSt.-Satz in Deutschland: 19% (0.19)

JSON SCHEMA:
{schema}

RECHNUNGSTEXT ZUR EXTRAKTION:
{invoice_text}

JSON AUSGABE:"""


ESTONIAN_EXTRACTION_PROMPT = """Sa oled tehisintellekti assistent, kes on spetsialiseerunud struktureeritud andmete eraldamisele Euroopa arvetest. Sinu ülesanne on eraldada teave esitatud arve tekstist ja tagastada see kehtiva JSON-ina, mis vastab esitatud skeemile.

KRIITILISED JUHISED:
- Tagasta AINULT kehtiv JSON, mitte lisateksti ega selgitusi
- Kasuta skeemist täpseid väljanimesid
- Teisenda kõik rahalised summad numbriteks (eemalda valuutasümbolid)
- Kasuta kõigi kuupäevade jaoks YYYY-MM-DD formaati
- Kui teave ei ole saadaval, kasuta null
- Kümnendkohtade jaoks kasuta punkte kümnendkoha eraldajatena
- Veendu, et kõik nõutud väljad on kaasatud

EESTI ARVE KONVENTSIOONID:
- Käibemaks = VAT (Value Added Tax)
- Kuupäev = issue_date
- Arve number = invoice_id
- Müüja = supplier
- Ostja = customer
- Netosumma = total_excl_vat
- Brutosumma = total_incl_vat
- Maksusumma = total_vat
- Standardne käibemaksumäär Eestis: 20% (0.20)
- Reg nr = vat_id

JSON SKEEM:
{schema}

ARVE TEKST ERALDAMISEKS:
{invoice_text}

JSON VÄLJUND:"""


def get_line_items_prompt(table_text: str, detected_language: str = "en") -> str:
    """
    Generate prompt specifically for line items extraction from tables

    Args:
        table_text: Text extracted from invoice tables
        detected_language: Detected language code

    Returns:
        Formatted prompt for line items extraction
    """
    prompts = {"en": LINE_ITEMS_ENGLISH_PROMPT, "de": LINE_ITEMS_GERMAN_PROMPT, "et": LINE_ITEMS_ESTONIAN_PROMPT}

    prompt_template = prompts.get(detected_language, prompts["en"])
    return prompt_template.format(table_text=table_text)


LINE_ITEMS_ENGLISH_PROMPT = """Extract invoice line items from the following table text. Return as JSON array of objects with fields: description, quantity, unit_price, line_total, vat_rate.

IMPORTANT:
- Convert all amounts to numbers
- Use null for missing values
- Ensure quantity × unit_price = line_total (approximately)
- VAT rates as decimals (20% = 0.20)

TABLE TEXT:
{table_text}

JSON ARRAY:"""


LINE_ITEMS_GERMAN_PROMPT = """Extrahiere Rechnungspositionen aus dem folgenden Tabellentext. Gib als JSON-Array von Objekten mit Feldern zurück: description, quantity, unit_price, line_total, vat_rate.

WICHTIG:
- Konvertiere alle Beträge in Zahlen
- Verwende null für fehlende Werte
- Stelle sicher, dass Menge × Einzelpreis = Gesamtpreis (ungefähr)
- MwSt.-Sätze als Dezimalzahlen (19% = 0.19)

TABELLENTEXT:
{table_text}

JSON ARRAY:"""


LINE_ITEMS_ESTONIAN_PROMPT = """Eralda arve read järgmisest tabeli tekstist. Tagasta JSON massiivina objektidest väljadega: description, quantity, unit_price, line_total, vat_rate.

TÄHTIS:
- Teisenda kõik summad numbriteks
- Kasuta null puuduvate väärtuste jaoks
- Veendu, et kogus × ühikuhind = rea kokku (ligikaudu)
- Käibemaksumäärad kümnendkohtadena (20% = 0.20)

TABELI TEKST:
{table_text}

JSON MASSIIV:"""


def get_correction_prompt(
    extracted_data: Dict[str, Any], validation_errors: list, detected_language: str = "en"
) -> str:
    """
    Generate prompt for correcting extraction errors

    Args:
        extracted_data: Previously extracted data with errors
        validation_errors: List of validation errors
        detected_language: Detected language code

    Returns:
        Formatted correction prompt
    """
    errors_text = "\n".join([f"- {error}" for error in validation_errors])

    prompts = {"en": CORRECTION_ENGLISH_PROMPT, "de": CORRECTION_GERMAN_PROMPT, "et": CORRECTION_ESTONIAN_PROMPT}

    prompt_template = prompts.get(detected_language, prompts["en"])

    return prompt_template.format(extracted_data=json.dumps(extracted_data, indent=2), errors=errors_text)


CORRECTION_ENGLISH_PROMPT = """The following invoice data has validation errors. Please correct the data and return valid JSON.

VALIDATION ERRORS:
{errors}

EXTRACTED DATA TO CORRECT:
{extracted_data}

CORRECTED JSON:"""


CORRECTION_GERMAN_PROMPT = """Die folgenden Rechnungsdaten haben Validierungsfehler. Bitte korrigiere die Daten und gib gültiges JSON zurück.

VALIDIERUNGSFEHLER:
{errors}

ZU KORRIGIERENDE EXTRAHIERTE DATEN:
{extracted_data}

KORRIGIERTES JSON:"""


CORRECTION_ESTONIAN_PROMPT = """Järgmistel arve andmetel on valideerimise vead. Palun paranda andmed ja tagasta kehtiv JSON.

VALIDEERIMISE VEAD:
{errors}

PARANDATAVAD ERALDATUD ANDMED:
{extracted_data}

PARANDATUD JSON:"""


def get_few_shot_examples(language: str = "en") -> str:
    """
    Get few-shot examples for better extraction performance

    Args:
        language: Language code for examples

    Returns:
        Few-shot examples text
    """
    examples = {"en": ENGLISH_EXAMPLES, "de": GERMAN_EXAMPLES, "et": ESTONIAN_EXAMPLES}

    return examples.get(language, examples["en"])


ENGLISH_EXAMPLES = """
EXAMPLE 1:
Invoice Text: "Invoice #INV-2024-001 Date: 15.01.2024 From: ABC Ltd, VAT: GB123456789 To: XYZ Corp Total: £120.00 VAT: £20.00 Net: £100.00"

JSON Output:
{
  "invoice_id": "INV-2024-001",
  "issue_date": "2024-01-15",
  "currency_code": "GBP",
  "supplier": {
    "name": "ABC Ltd",
    "vat_id": "GB123456789",
    "country_code": "GB"
  },
  "customer": {
    "name": "XYZ Corp",
    "vat_id": null,
    "country_code": null
  },
  "total_excl_vat": 100.00,
  "total_vat": 20.00,
  "total_incl_vat": 120.00,
  "invoice_lines": []
}
"""


GERMAN_EXAMPLES = """
BEISPIEL 1:
Rechnungstext: "Rechnung Nr. RE-2024-001 Datum: 15.01.2024 Von: Musterfirma GmbH, USt-IdNr: DE123456789 An: Kunde AG Gesamt: 119,00 € MwSt: 19,00 € Netto: 100,00 €"

JSON Ausgabe:
{
  "invoice_id": "RE-2024-001",
  "issue_date": "2024-01-15",
  "currency_code": "EUR",
  "supplier": {
    "name": "Musterfirma GmbH",
    "vat_id": "DE123456789",
    "country_code": "DE"
  },
  "customer": {
    "name": "Kunde AG",
    "vat_id": null,
    "country_code": null
  },
  "total_excl_vat": 100.00,
  "total_vat": 19.00,
  "total_incl_vat": 119.00,
  "invoice_lines": []
}
"""


ESTONIAN_EXAMPLES = """
NÄIDE 1:
Arve tekst: "Arve nr A-2024-001 Kuupäev: 15.01.2024 Müüja: Näidisfirma OÜ, Reg nr: EE123456789 Ostja: Klient AS Kokku: 120,00 € Käibemaks: 20,00 € Summa ilma km: 100,00 €"

JSON väljund:
{
  "invoice_id": "A-2024-001",
  "issue_date": "2024-01-15",
  "currency_code": "EUR",
  "supplier": {
    "name": "Näidisfirma OÜ",
    "vat_id": "EE123456789",
    "country_code": "EE"
  },
  "customer": {
    "name": "Klient AS",
    "vat_id": null,
    "country_code": null
  },
  "total_excl_vat": 100.00,
  "total_vat": 20.00,
  "total_incl_vat": 120.00,
  "invoice_lines": []
}
"""


def get_confidence_prompt(extracted_data: Dict[str, Any], detected_language: str = "en") -> str:
    """
    Generate prompt for confidence scoring of extracted data

    Args:
        extracted_data: Extracted invoice data
        detected_language: Detected language code

    Returns:
        Formatted confidence scoring prompt
    """
    prompts = {"en": CONFIDENCE_ENGLISH_PROMPT, "de": CONFIDENCE_GERMAN_PROMPT, "et": CONFIDENCE_ESTONIAN_PROMPT}

    prompt_template = prompts.get(detected_language, prompts["en"])

    return prompt_template.format(extracted_data=json.dumps(extracted_data, indent=2))


CONFIDENCE_ENGLISH_PROMPT = """Rate the confidence of this invoice data extraction on a scale of 0-1 where:
- 1.0 = Perfect extraction, all data consistent and complete
- 0.8-0.9 = Good extraction, minor missing fields
- 0.6-0.7 = Adequate extraction, some inconsistencies
- 0.4-0.5 = Poor extraction, major issues
- 0.0-0.3 = Very poor extraction, mostly incorrect

Consider:
- Data completeness
- Internal consistency (totals match calculations)
- Format correctness
- Logical consistency

EXTRACTED DATA:
{extracted_data}

Return only a number between 0 and 1:"""


CONFIDENCE_GERMAN_PROMPT = """Bewerte das Vertrauen in diese Rechnungsdatenextraktion auf einer Skala von 0-1, wobei:
- 1.0 = Perfekte Extraktion, alle Daten konsistent und vollständig
- 0.8-0.9 = Gute Extraktion, kleinere fehlende Felder
- 0.6-0.7 = Angemessene Extraktion, einige Inkonsistenzen
- 0.4-0.5 = Schlechte Extraktion, größere Probleme
- 0.0-0.3 = Sehr schlechte Extraktion, meist inkorrekt

Berücksichtige:
- Datenvollständigkeit
- Interne Konsistenz (Summen stimmen mit Berechnungen überein)
- Formatkorrektheit
- Logische Konsistenz

EXTRAHIERTE DATEN:
{extracted_data}

Gib nur eine Zahl zwischen 0 und 1 zurück:"""


CONFIDENCE_ESTONIAN_PROMPT = """Hinda selle arve andmete eraldamise usaldusväärsust skaalal 0-1, kus:
- 1.0 = Täiuslik eraldamine, kõik andmed järjepidevad ja täielikud
- 0.8-0.9 = Hea eraldamine, väiksed puuduvad väljad
- 0.6-0.7 = Piisav eraldamine, mõned ebajärjepidevused
- 0.4-0.5 = Halb eraldamine, suuremad probleemid
- 0.0-0.3 = Väga halb eraldamine, enamasti vale

Kaaluda:
- Andmete täielikkus
- Sisemine järjepidevus (kogusummad vastavad arvutustele)
- Formaadi õigsus
- Loogiline järjepidevus

ERALDATUD ANDMED:
{extracted_data}

Tagasta ainult arv 0 ja 1 vahel:"""
