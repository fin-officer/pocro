# Instrukcja instalacji

Ten przewodnik przeprowadzi Cię przez proces konfiguracji systemu do przetwarzania faktur z wykorzystaniem OCR i LLM.

## Spis treści
- [Wymagania wstępne](#wymagania-wstępne)
- [Szybki start](#szybki-start)
- [Instalacja krok po kroku](#instalacja-krok-po-kroku)
- [Konfiguracja](#konfiguracja)
- [Weryfikacja instalacji](#weryfikacja-instalacji)
- [Rozwiązywanie problemów](#rozwiązywanie-problemów)
- [Dodatkowe zasoby](#dodatkowe-zasoby)

> **Uwaga:** W razie problemów z instalacją, zapoznaj się z [przewodnikiem rozwiązywania problemów](./TROUBLESHOOTING.md).

## Wymagania wstępne

- Python 3.10+
- pip (menedżer pakietów Pythona)
- (Opcjonalnie) Karta graficzna z obsługą CUDA dla lepszej wydajności
- (Opcjonalnie) Docker i Docker Compose do uruchomienia w kontenerze
- (Opcjonalnie) Połączenie z internetem do pobrania modeli

## Szybki start

1. **Sklonuj repozytorium**
   ```bash
   git clone git@github.com:fin-officer/pocro.git
   cd pocro
   ```

2. **Skonfiguruj środowisko**
   ```bash
   # Skopiuj przykładowy plik środowiskowy
   cp .env.example .env
   ```

   Zmodyfikuj plik `.env` zgodnie ze swoimi potrzebami. Minimalna konfiguracja:
   ```env
   MODEL_NAME=facebook/opt-125m  # lub inny preferowany model
   OCR_ENGINE=easyocr  # lub paddleocr
   LOG_LEVEL=INFO
   ```

3. **Zainstaluj zależności**
   ```bash
   # Utwórz i aktywuj środowisko wirtualne (zalecane)
   python -m venv .venv
   source .venv/bin/activate  # Na Windows: .venv\Scripts\activate

   # Zainstaluj zależności za pomocą Poetry
   poetry install  # Dla środowiska developerskiego
   # lub
   poetry install --no-dev  # Tylko zależności produkcyjne
   ```

4. **Uruchom aplikację**
   ```bash
   make run  # Używa Uvicorn z automatycznym przeładowaniem
   # lub ręcznie:
   # uvicorn src.main:app --host 0.0.0.0 --port 8088 --reload
   ```

   API będzie dostępne pod adresem: `http://localhost:8088`
   
   Sprawdź działanie:
   ```bash
   curl http://localhost:8088/health
   ```

## Konfiguracja

### Zmienne środowiskowe

Kluczowe opcje konfiguracyjne w pliku `.env`:

#### Konfiguracja modelu LLM
- `MODEL_NAME` - Nazwa modelu z Hugging Face (np. `facebook/opt-125m`)
- `QUANTIZATION` - Metoda kwantyzacji modelu (domyślnie: `awq`)
- `MAX_MODEL_LENGTH` - Maksymalna długość generowanego tekstu (domyślnie: `2048`)
- `TEMPERATURE` - Parametr temperatury generowania

### Przetwarzanie faktur

Aby przetworzyć fakturę, użyj komendy `pocro`:

```bash
# Przetwarzanie pojedynczej faktury
pocro process ścieżka/do/faktura.pdf

# Z określonym plikiem wyjściowym
pocro process ścieżka/do/faktura.pdf --output wynik.json

# Wyświetlenie pomocy
pocro --help
pocro process --help
```

Dostępne opcje dla komendy `process`:
- `--output`, `-o`: Określa plik wyjściowy (domyślnie: `<nazwa_pliku_wejsciowego>.json`)
- `--format`, `-f`: Format wyjściowy (`json` lub `csv`, domyślnie: `json`)

#### Konfiguracja serwera
- `HOST` - Adres, na którym ma działać serwer (domyślnie: `0.0.0.0`)
- `PORT` - Port serwera (domyślnie: `8088`)
- `DEBUG` - Tryb debugowania (domyślnie: `False`)
- `LOG_LEVEL` - Poziom logowania (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `CACHE_DIR` - Katalog na cache modeli (domyślnie: `./.cache`)

#### Przykładowa konfiguracja
```env
# Konfiguracja modelu
MODEL_NAME=facebook/opt-125m
QUANTIZATION=awq
MAX_MODEL_LENGTH=2048

# Konfiguracja OCR
OCR_ENGINE=easyocr
OCR_LANGUAGES=pl,en,de

# Konfiguracja serwera
HOST=0.0.0.0
PORT=8088
DEBUG=False
LOG_LEVEL=INFO
CACHE_DIR=./.cache
```

## Instalacja z Dockerem

1. **Zbuduj obraz Dockera**
   ```bash
   make docker-build
   ```
   
   lub ręcznie:
   ```bash
   docker build -t pocro .
   ```

2. **Uruchom kontener**
   ```bash
   make docker-run
   ```
   
   lub ręcznie:
   ```bash
   docker run -d --name pocro -p 8088:8088 --env-file .env pocro
   ```

3. **Zatrzymaj kontener**
   ```bash
   make docker-stop
   ```

## Weryfikacja instalacji

1. **Sprawdź stan API**
   ```bash
   curl http://localhost:8088/health
   ```
   Powinno zwrócić: `{"status":"ok"}`

2. **Przetestuj rozpoznawanie faktury**
   ```bash
   curl -X POST http://localhost:8088/process \
     -H "Content-Type: application/json" \
     -d '{
       "invoice_number": "FV/2023/1234",
       "issue_date": "2023-05-15",
       "due_date": "2023-06-14",
       "total_amount": 1234.56,
       "currency": "PLN",
       "supplier": {
         "name": "Przykładowy Dostawca Sp. z o.o.",
         "tax_id": "1234567890"
       },
       "customer": {
         "name": "Klient Przykładowy",
         "tax_id": "0987654321"
       },
       "items": [
         {
           "description": "Usługa przykładowa",
           "quantity": 1,
           "unit_price": 1000.00,
           "tax_rate": 23,
           "amount": 1230.00
         }
       ]
     }'
   ```

## Rozwiązywanie problemów

### Typowe problemy i rozwiązania

- **Brak modułów Pythona**
  ```bash
  # Zainstaluj brakujące zależności
  poetry install
  ```

- **Błąd ładowania modelu**
  - Sprawdź, czy `MODEL_NAME` w `.env` wskazuje na poprawny model
  - Upewnij się, że masz dostęp do internetu do pobrania modelu
  - Sprawdź uprawnienia do katalogu cache (domyślnie `~/.cache/`)

- **Błędy CUDA**
  - Upewnij się, że masz zainstalowane odpowiednie sterowniki NVIDIA
  - Sprawdź zgodność wersji CUDA z wymaganiami bibliotek
  - Spróbuj wyłączyć akcelerację GPU ustawiając `OCR_GPU=False`

- **Port zajęty**
  - Zmień port w zmiennej `PORT` w pliku `.env`
  - Sprawdź, czy inna aplikacja nie używa tego samego portu: `lsof -i :8088`

- **Problemy z zależnościami**
  ```bash
  # Wyczyść środowisko i zainstaluj ponownie
  rm -rf .venv
  python -m venv .venv
  source .venv/bin/activate
  poetry install
  ```

W przypadku dalszych problemów, zapoznaj się z przewodnikiem [Rozwiązywanie problemów](./TROUBLESHOOTING.md).

## Następne kroki

- [Dokumentacja API](./API_DOCUMENTATION.md)
- [Przewodnik wdrażania](./DEPLOYMENT.md)
- [Rozwiązywanie problemów](./TROUBLESHOOTING.md)