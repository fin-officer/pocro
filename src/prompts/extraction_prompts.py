"""
Prompt templates for multilingual invoice extraction
"""
import json
from typing import Dict

from ..models.invoice_schema import InvoiceData


def create_multilingual_prompt(invoice_text: str, detected_language: str = 'en') -> str:
    """Create multilingual prompt for invoice extraction"""

    prompts = {
        'de': GERMAN_EXTRACTION_PROMPT,
        'en': ENGLISH_EXTRACTION_PROMPT,
        'et': ESTONIAN_EXTRACTION_PROMPT
    }

    template = prompts.get(detected_language, prompts['en'])
    schema = json.dumps(InvoiceData.model_json_schema(), indent=2)

    return template.format(
        schema=schema,
        invoice_text=invoice_text
    )


GERMAN_EXTRACTION_PROMPT = """Du bist ein KI-Assistent, der auf die Extraktion strukturierter Daten aus deutschen Rechnungen spezialisiert ist.

Extrahiere die folgenden Informationen aus dem Rechnungstext und gib sie als gültiges JSON zurück, das dem bereitgestellten Schema entspricht.

WICHTIGE REGELN:
- Gib NUR gültiges JSON zurück, keinen zusätzlichen Text
- Verwende die exakten Feldnamen aus dem Schema
- Konvertiere alle Beträge in Zahlen (keine Währungssymbole)
- Verwende das Format YYYY-MM-DD für Daten
- Wenn Informationen nicht verfügbar sind, verwende null
- Erkenne MwSt./USt./Mehrwertsteuer als VAT
- Deutsche Dezimaltrennzeichen (,) in Punkt (.) umwandeln

Schema:
{schema}

Rechnungstext:
{invoice_text}

JSON Output:"""

ENGLISH_EXTRACTION_PROMPT = """You are an AI assistant specialized in extracting structured data from English invoices.

Extract the following information from the invoice text and return it as valid JSON matching the provided schema.

IMPORTANT RULES:
- Return ONLY valid JSON, no additional text
- Use exact field names from the schema
- Convert all amounts to numbers (no currency symbols)
- Use YYYY-MM-DD format for dates
- If information is not available, use null
- Recognize VAT/Tax as tax information

Schema:
{schema}

Invoice Text:
{invoice_text}

JSON Output:"""

ESTONIAN_EXTRACTION_PROMPT = """Sa oled tehisintellekti assistent, kes on spetsialiseerunud eesti arvete struktureeritud andmete eraldamisele.

Eralda järgmine teave arve tekstist ja tagasta see kehtiva JSON-ina, mis vastab esitatud skeemile.

TÄHTIS:
- Tagasta AINULT kehtiv JSON, mitte lisateksti
- Kasuta skeemist täpseid väljanimesid
- Teisenda kõik summad numbriteks (mitte valuutasümboleid)
- Kasuta kuupäevade jaoks YYYY-MM-DD formaati
- Kui teave ei ole saadaval, kasuta null
- Käibemaks = VAT

Skeem:
{schema}

Arve tekst:
{invoice_text}

JSON väljund:"""