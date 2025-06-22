# Dokumentacja API

Ten dokument zawiera szczegółowe informacje o dostępnych endpointach API do przetwarzania faktur, formatach żądań i odpowiedzi oraz przykłady użycia.

## Spis treści
- [Podstawowy adres URL](#podstawowy-adres-url)
- [Uwierzytelnianie](#uwierzytelnianie)
- [Endpointy](#endpointy)
  - [Sprawdzenie stanu](#sprawdzenie-stanu)
  - [Przetwarzanie faktury](#przetwarzanie-faktury)
  - [Pobieranie wyników](#pobieranie-wyników)
- [Kody błędów](#kody-błędów)
- [Przykłady użycia](#przykłady-użycia)
- [Ograniczenia i limity](#ograniczenia-i-limity)
- [Rozwiązywanie problemów](#rozwiązywanie-problemów)
- [Więcej informacji](#więcej-informacji)

> **Uwaga:** W razie problemów z API, zapoznaj się z [przewodnikiem rozwiązywania problemów](./TROUBLESHOOTING.md).

## Podstawowy adres URL

All API endpoints are relative to the base URL:
```
http://localhost:8088  # or your configured host:port
```

## Authentication

Currently, the API does not require authentication for local development. For production use, consider implementing API key authentication.

## Endpoints

### Health Check

Check if the API is running.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "ok"
}
```

### Process Invoice

Process an invoice image or PDF and extract structured data.

**Endpoint**: `POST /process`

**Content-Type**: `multipart/form-data`

**Request Parameters**:
- `file` (required): The invoice file (image or PDF)
- `extract_tables` (optional, boolean): Whether to extract tables (default: `true`)
- `language` (optional, string): Language code (e.g., 'en', 'de', 'et')

**Example Request**:
```bash
curl -X POST http://localhost:8088/process \
  -F "file=@/path/to/invoice.pdf" \
  -F "extract_tables=true" \
  -F "language=en"
```

**Success Response (200 OK)**:
```json
{
  "status": "success",
  "data": {
    "vendor_name": "Example Corp",
    "invoice_number": "INV-2023-001",
    "invoice_date": "2023-01-15",
    "due_date": "2023-02-14",
    "total_amount": 1234.56,
    "currency": "EUR",
    "line_items": [
      {
        "description": "Product A",
        "quantity": 2,
        "unit_price": 500.00,
        "total": 1000.00
      }
    ],
    "tables": [
      {
        "header": ["Description", "Qty", "Price", "Total"],
        "rows": [
          ["Product A", "2", "500.00", "1000.00"]
        ]
      }
    ],
    "metadata": {
      "processing_time": 1.23,
      "model": "facebook/opt-125m",
      "ocr_engine": "easyocr"
    }
  }
}
```

**Error Response (400 Bad Request)**:
```json
{
  "status": "error",
  "error": {
    "code": "invalid_file",
    "message": "Unsupported file type. Please upload a PDF or image file."
  }
}
```

### Process Text

Process raw invoice text and extract structured data.

**Endpoint**: `POST /process/text`

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "text": "INVOICE\nVendor: Example Corp\nInvoice #: INV-2023-001\n...",
  "language": "en"
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8088/process/text \
  -H "Content-Type: application/json" \
  -d '{"text":"INVOICE\\nVendor: Example Corp\\n...", "language":"en"}'
```

**Response**:
Same structure as the `/process` endpoint.

## Rate Limiting

By default, the API is limited to 60 requests per minute per IP address. If exceeded, you'll receive a 429 Too Many Requests response.

## Error Handling

All error responses follow this format:

```json
{
  "status": "error",
  "error": {
    "code": "error_code",
    "message": "Human-readable error message"
  }
}
```

Common error codes:
- `invalid_file`: Unsupported or corrupted file
- `file_too_large`: File exceeds maximum allowed size (default: 50MB)
- `invalid_request`: Missing or invalid parameters
- `processing_error`: Error during invoice processing
- `rate_limit_exceeded`: Too many requests

## Testing the API

You can test the API using the built-in Swagger UI:

1. Start the application
2. Open `http://localhost:8088/docs` in your browser
3. Use the interactive UI to test endpoints

## Client Libraries

### Python Example

```python
import requests

def process_invoice(api_url, file_path, language='en'):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'language': language}
        response = requests.post(f"{api_url}/process", files=files, data=data)
        return response.json()

# Usage
result = process_invoice('http://localhost:8088', 'invoice.pdf')
print(result)
```

## Versioning

API versioning is not yet implemented. All endpoints are currently under `/v1/`.

## Support

For support, please open an issue on our [GitHub repository](https://github.com/finofficer/pocro/issues).