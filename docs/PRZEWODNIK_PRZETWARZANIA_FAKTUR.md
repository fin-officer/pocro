# Przewodnik przetwarzania faktur z PDF do JSON przy użyciu OCR

Ten dokument zawiera szczegółową instrukcję krok po kroku, jak przetwarzać faktury z formatu PDF do struktury JSON przy użyciu systemu OCR.

## Wymagania wstępne

1. Zainstalowany Python 3.10
2. Zainstalowane zależności projektu (patrz [INSTALLATION.md](./INSTALLATION.md))
3. Dostęp do systemu z zainstalowanym Tesseract OCR (wymagane do przetwarzania obrazów)
4. Przykładowy plik faktury w formacie PDF

## Krok 1: Przygotowanie środowiska

Upewnij się, że masz aktywowane środowisko wirtualne i zainstalowane wszystkie zależności:

```bash
# Aktywacja środowiska wirtualnego
source .venv/bin/activate  # Linux/macOS
# lub
.venv\Scripts\activate    # Windows

# Zainstaluj zależności używając Poetry:
poetry install
```

## Krok 2: Przygotowanie pliku faktury

1. Umieść plik faktury w formacie PDF w katalogu `data/input/`
2. Upewnij się, że plik ma uprawnienia do odczytu

## Krok 3: Uruchomienie procesu przetwarzania

### Opcja 1: Użycie skryptu CLI

```bash
python -m src.cli.process_invoice --input data/input/faktura.pdf --output data/output/faktura.json
```

### Opcja 2: Użycie API

1. Uruchom serwer API:
   ```bash
   uvicorn src.api.main:app --reload
   ```

2. Wyślij żądanie POST do endpointu przetwarzania faktur:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/invoice/process" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@data/input/faktura.pdf" \
        -o data/output/faktura.json
   ```

## Krok 4: Weryfikacja wyników

Sprawdź wygenerowany plik JSON w katalogu `data/output/`. Powinien zawierać rozpoznane dane z faktury w ustrukturyzowanym formacie, np.:

```json
{
  "invoice_number": "FV/2023/123",
  "issue_date": "2023-06-22",
  "due_date": "2023-07-22",
  "seller": {
    "name": "Przykładowa Firma Sp. z o.o.",
    "tax_id": "1234567890",
    "address": "ul. Przykładowa 123, 00-001 Warszawa"
  },
  "buyer": {
    "name": "Klient Sp. z o.o.",
    "tax_id": "0987654321",
    "address": "ul. Testowa 456, 30-001 Kraków"
  },
  "items": [
    {
      "name": "Usługa programistyczna",
      "quantity": 1,
      "unit": "szt",
      "unit_price_net": 100.00,
      "tax_rate": 23,
      "net_amount": 100.00,
      "gross_amount": 123.00
    }
  ],
  "total_net_amount": 100.00,
  "total_tax_amount": 23.00,
  "total_gross_amount": 123.00,
  "currency": "PLN",
  "payment_method": "przelew",
  "bank_account": "12 3456 7890 1234 5678 9012 3456"
}
```

## Typowe problemy i ich rozwiązania

### 1. Niska jakość rozpoznawania tekstu

**Objawy**:
- Błędnie rozpoznane znaki
- Brakujące fragmenty tekstu
- Błędne rozpoznanie czcionek

**Rozwiązania**:
- Upewnij się, że skan/PDF jest dobrej jakości
- Spróbuj zwiększyć rozdzielczość obrazu przed przetwarzaniem
- Sprawdź ustawienia OCR (język, tryb przetwarzania)
- W przypadku faktur z tabelami, upewnij się, że struktura tabeli jest poprawnie rozpoznawana

### 2. Błędne rozpoznawanie dat i liczb

**Objawy**:
- Błędnie rozpoznane daty (np. 01.01.2023 → 0l.0l.2O23)
- Błędnie rozpoznane kwoty (np. 100,00 → 100.00)

**Rozwiązania**:
- Sprawdź ustawienia regionalne (locale) systemu
- Wykorzystaj walidację i korektę rozpoznanych wartości
- Rozważ użycie wyrażeń regularnych do ekstrakcji dat i liczb

### 3. Brak rozpoznania niektórych pól

**Objawy**:
- Brakujące wymagane pola w wyjściowym JSON
- Błędnie zmapowane wartości

**Rozwiązania**:
- Sprawdź logi przetwarzania w poszukiwaniu błędów
- Dostosuj szablony rozpoznawania do formatu faktury
- Rozważ dodanie ręcznej weryfikacji dla krytycznych pól

### 4. Problemy z wydajnością

**Objawy**:
- Długi czas przetwarzania
- Wysokie zużycie pamięci

**Rozwiązania**:
- Zoptymalizuj rozmiar przetwarzanych obrazów
- Rozważ przetwarzanie wsadowe dla wielu faktur
- Sprawdź ustawienia buforowania i przetwarzania równoległego

## Zaawansowane użycie

### Przetwarzanie wsadowe

Aby przetworzyć wiele faktur na raz:

```bash
python -m src.cli.batch_process \
    --input-dir data/input/invoices/ \
    --output-dir data/output/processed/
```

### Dostosowywanie modeli OCR

Możesz dostosować parametry OCR w pliku konfiguracyjnym `config/ocr_config.yaml`:

```yaml
tesseract:
  lang: pol+eng  # Języki do rozpoznawania
  psm: 6         # Tryb segmentacji strony
  oem: 3         # Tryb silnika OCR
  dpi: 300       # Rozdzielczość przetwarzania

preprocessing:
  deskew: true   # Korekcja pochylenia obrazu
  denoise: true  # Redukcja szumów
  enhance: true  # Poprawa jakości obrazu
```

### Integracja z systemami zewnętrznymi

Przykład integracji z systemem ERP:

```python
import requests
import json

def process_and_send_invoice(pdf_path, api_url, api_key):
    with open(pdf_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{api_url}/api/v1/invoice/process",
            files=files,
            headers={'X-API-KEY': api_key}
        )
    
    if response.status_code == 200:
        invoice_data = response.json()
        # Zapisz dane do bazy danych lub wyślij do systemu ERP
        with open('processed_invoice.json', 'w') as f:
            json.dump(invoice_data, f, indent=2, ensure_ascii=False)
        return True
    else:
        print(f"Błąd przetwarzania faktury: {response.text}")
        return False
```

## Wsparcie

W przypadku problemów z przetwarzaniem faktur:
1. Sprawdź logi aplikacji w katalogu `logs/`
2. Sprawdź [przewodnik rozwiązywania problemów](./TROUBLESHOOTING.md)
3. Sprawdź istniejące zgłoszenia lub utwórz nowe w [repozytorium projektu](https://github.com/finofficer/pocro/issues)

## Licencja

Ten przewodnik jest dostępny na licencji MIT. Szczegóły w pliku LICENSE w głównym katalogu projektu.
