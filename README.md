# pocro

System do automatycznego przetwarzania faktur z wykorzystaniem OCR i LLM.

## Kluczowe funkcjonalności:

✅ **Pełny stack OCR+LLM** - EasyOCR/PaddleOCR + Mistral/Qwen/LLaMA  
✅ **GPU optimized** - AWQ/NF4 quantization dla 8GB VRAM  
✅ **Multilingual** - DE/EN/EE z automatyczną detekcją języka  
✅ **EU compliance** - Schema EN 16931 i PEPPOL  
✅ **Production ready** - Docker, monitoring, health checks  
✅ **Comprehensive tests** - Unit, integration, performance  
✅ **Easy deployment** - Makefile, scripts, dokumentacja  
✅ **Flexible configuration** - Environment variables and Pydantic settings  
✅ **Robust error handling** - Fallback mechanisms and detailed logging  

## Wymagania systemowe

- Python 3.10+
- Docker, docker-compose (do uruchomienia w kontenerze)
- (Opcjonalnie) GPU z min. 8GB VRAM dla modeli LLM
- (Opcjonalnie) Połączenie z internetem do pobrania modeli

## Szybki start:

```bash
# 1. Klonowanie i setup
git clone git@github.com:fin-officer/pocro.git
cd pocro
cp .env.example .env

# 2. Instalacja zależności produkcyjnych
make install

# 3. (Opcjonalnie) Instalacja zależności developerskich
make install-dev

# 4. Setup środowiska (modele, uprawnienia, pre-commit)
make setup

# 5. Pobranie modeli OCR/LLM
make download-models

# 6. Uruchomienie lokalnie
make run

# 7. Lub z Docker
make docker-build
make docker-run

# 8. Testy
make test
make test-cov

# 9. Lintowanie i formatowanie
make lint
make format

# 10. Walidacja instalacji
make validate

# 11. Benchmark
make benchmark
```

## Struktura projektu

```
pocro/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── docker-compose.dev.yml
├── Makefile
├── setup.py
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   ├── config/
│   ├── core/
│   ├── models/
│   ├── prompts/
│   └── utils/
│
├── scripts/
│   ├── benchmark.py
│   ├── download_models.py
│   ├── migrate_data.py
│   ├── setup_environment.sh
│   └── validate_installation.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/
│   ├── integration/
│   ├── performance/
│   ├── test_main.py
│   └── unit/
│
├── configs/
│   ├── deployment/
│   ├── model_configs/
│   └── ocr_configs/
│
├── monitoring/
│   ├── __init__.py
│   ├── dashboards/
│   ├── health_check.py
│   └── metrics.py
│
├── data/
├── logs/
├── docs/
└── venv/
```

- **src/** – kod źródłowy aplikacji (API, core, modele, konfiguracje, prompty, utils)
- **scripts/** – skrypty narzędziowe: setup, walidacja, pobieranie modeli, migracje, benchmark
- **tests/** – testy jednostkowe, integracyjne, performance, fixtures
- **configs/** – konfiguracje deploymentu, modeli, OCR
- **monitoring/** – monitoring, health-check, metryki, dashboardy
- **data/** – dane wejściowe/wyjściowe
- **logs/** – logi aplikacji
- **docs/** – dokumentacja

## Najważniejsze komendy Makefile

### Środowisko i zależności
- `make install` – instalacja zależności produkcyjnych
- `make install-dev` – instalacja zależności developerskich + pre-commit
- `make setup` – pełny setup środowiska
- `make clean` – czyszczenie środowiska
- `make download-models` – pobranie modeli OCR/LLM

### Uruchamianie
- `make run` – uruchomienie lokalnie (FastAPI/Uvicorn)
- `make docker-build` – budowa obrazu Docker
- `make docker-run` – uruchomienie Dockera
- `make docker-dev` – uruchomienie środowiska developerskiego w Dockerze
- `make docker-stop` – zatrzymanie kontenerów Docker

### Testowanie i jakość kodu
- `make test` – testy jednostkowe i integracyjne
- `make test-cov` – testy z pokryciem kodu
- `make lint` – lintowanie (flake8, mypy, black, isort)
- `make format` – autoformatowanie kodu
- `make validate` – walidacja instalacji
- `make benchmark` – benchmark modeli/algorytmów

### Pomocnicze
- `make help` – wyświetla dostępne komendy
- `make check-env` – sprawdza poprawność środowiska
- `make update-deps` – aktualizuje zależności do najnowszych wersji

- `make install` – instalacja zależności produkcyjnych
- `make install-dev` – instalacja zależności developerskich + pre-commit
- `make setup` – pełny setup środowiska
- `make run` – uruchomienie lokalnie (FastAPI/Uvicorn)
- `make docker-build` – budowa obrazu Docker
- `make docker-run` – uruchomienie Dockera
- `make docker-dev` – uruchomienie środowiska developerskiego w Dockerze
- `make docker-stop` – zatrzymanie kontenerów Docker
- `make test` – testy
- `make test-cov` – testy z pokryciem kodu
- `make lint` – lintowanie (flake8, mypy, black, isort)
- `make format` – autoformatowanie kodu
- `make clean` – czyszczenie środowiska
- `make validate` – walidacja instalacji
- `make download-models` – pobranie modeli OCR/LLM
- `make benchmark` – benchmark modeli/algorytmów

## Konfiguracja środowiska

### Plik .env

Skopiuj przykładową konfigurację i dostosuj do swoich potrzeb:

```bash
cp .env.example .env
```

Kluczowe zmienne środowiskowe:

- `MODEL_NAME` - nazwa modelu LLM (domyślnie: `facebook/opt-125m`)
- `OCR_ENGINE` - silnik OCR do użycia (`easyocr` lub `paddleocr`)
- `LOG_LEVEL` - poziom logowania (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `CACHE_DIR` - katalog na cache modeli

### Zarządzanie zależnościami

Projekt używa Poetry do zarządzania zależnościami. Zainstaluj zależności:

```bash
poetry install  # Dla środowiska developerskiego
poetry install --no-dev  # Tylko zależności produkcyjne
```

## Testowanie

Uruchom pełną baterię testów:

```bash
make test  # Testy jednostkowe i integracyjne
make test-cov  # Testy z pokryciem kodu
make lint  # Sprawdzenie jakości kodu
```

## Dokumentacja

Szczegółowa dokumentacja dostępna jest w katalogu `docs/`:

- [Instalacja i konfiguracja](docs/INSTALLATION.md)
- [Dokumentacja API](docs/API_DOCUMENTATION.md)
- [Wdrażanie](docs/DEPLOYMENT.md)
- [Rozwiązywanie problemów](docs/TROUBLESHOOTING.md)

---

Masz pytania lub chcesz zgłosić błąd? Otwórz issue lub napisz na contact@finofficer.com
