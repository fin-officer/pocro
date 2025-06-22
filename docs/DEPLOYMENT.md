# Przewodnik wdrażania

Ten dokument zawiera szczegółowe instrukcje dotyczące wdrażania systemu do przetwarzania faktur w różnych środowiskach.

## Spis treści
- [Wymagania wstępne](#wymagania-wstępne)
- [Wdrażanie lokalne](#wdrażanie-lokalne)
- [Wdrażanie w kontenerze Docker](#wdrażanie-w-kontenerze-docker)
- [Wdrażanie w chmurze](#wdrażanie-w-chmurze)
- [Konfiguracja produkcyjna](#konfiguracja-produkcyjna)
- [Skalowanie](#skalowanie)
- [Monitorowanie](#monitorowanie)
- [Kopie zapasowe i przywracanie](#kopie-zapasowe-i-przywracanie)

## Wymagania wstępne

Przed rozpoczęciem wdrażania upewnij się, że masz zainstalowane następujące narzędzia:

- [Docker](https://docs.docker.com/get-docker/) (opcjonalnie, ale zalecane)
- [Docker Compose](https://docs.docker.com/compose/install/) (opcjonalnie, ale zalecane)
- [Python 3.10+](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation) (do zarządzania zależnościami)
- [Git](https://git-scm.com/downloads)

## Wdrażanie lokalne

### 1. Sklonuj repozytorium

```bash
git clone https://github.com/finofficer/pocro.git
cd pocro
```

### 2. Skonfiguruj środowisko

```bash
# Utwórz i aktywuj środowisko wirtualne
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# lub
.venv\Scripts\activate    # Windows

# Zainstaluj zależności
poetry install

# Skopiuj przykładowy plik konfiguracyjny
cp .env.example .env
```

### 3. Skonfiguruj zmienne środowiskowe

Edytuj plik `.env` i ustaw odpowiednie wartości:

```env
# Podstawowa konfiguracja
APP_ENV=production
DEBUG=False
SECRET_KEY=twoj_tajny_klucz

# Konfiguracja bazy danych
DATABASE_URL=postgresql://user:password@localhost:5432/invoice_db

# Konfiguracja modelu AI
MODEL_NAME=facebook/opt-125m
DEVICE=cuda  # lub cpu, jeśli nie masz karty graficznej

# Konfiguracja API
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Bezpieczeństwo
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

### 4. Zainicjuj bazę danych

```bash
# Uruchom migracje
alembic upgrade head

# Załaduj dane testowe (opcjonalnie)
python -m scripts.load_sample_data
```

### 5. Uruchom serwer deweloperski

```bash
uvicorn src.main:app --reload
```

Aplikacja będzie dostępna pod adresem: http://localhost:8000

## Wdrażanie w kontenerze Docker

### 1. Zbuduj obrazy

```bash
docker-compose -f docker-compose.prod.yml build
```

### 2. Uruchom kontenery

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Sprawdź logi

```bash
docker-compose -f docker-compose.prod.yml logs -f
```

## Wdrażanie w chmurze

### Opcja 1: AWS ECS

1. Zainstaluj i skonfiguruj AWS CLI:
   ```bash
   aws configure
   ```

2. Wdróż stos CloudFormation:
   ```bash
   aws cloudformation deploy \
     --template-file cloudformation/template.yaml \
     --stack-name invoice-processor \
     --capabilities CAPABILITY_IAM
   ```

### Opcja 2: Google Cloud Run

1. Zaloguj się do Google Cloud:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. Wdróż usługę:
   ```bash
   gcloud run deploy invoice-processor \
     --source . \
     --platform managed \
     --region europe-west3 \
     --allow-unauthenticated
   ```

## Konfiguracja produkcyjna

### Bezpieczeństwo

1. **TLS/SSL**: Skonfiguruj certyfikat SSL (np. Let's Encrypt)
2. **Uwierzytelnianie**: Włącz uwierzytelnianie JWT
3. **CORS**: Ogranicz źródła do zaufanych domen
4. **Rate limiting**: Włącz ograniczanie liczby żądań

### Wydajność

1. **Buforowanie**: Skonfiguruj Redis do buforowania
2. **Kolejkowanie**: Użyj Celery do zadań w tle
3. **Optymalizacja modelu**: Użyj skompilowanej wersji modelu

## Skalowanie

### Poziome skalowanie

```bash
# Przykład skalowania w Kubernetes
kubectl scale deployment invoice-processor --replicas=5
```

### Skalowanie automatyczne

```yaml
# Przykładowa konfiguracja HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: invoice-processor
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: invoice-processor
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Monitorowanie

### Metryki

Aplikacja eksportuje metryki w formacie Prometheus pod adresem `/metrics`.

### Alerty

Skonfiguruj alerty dla:
- Wysokiego wykorzystania CPU/pamięci
- Wysokiego czasu odpowiedzi
- Wysokiej liczby błędów

## Kopie zapasowe i przywracanie

### Baza danych

```bash
# Tworzenie kopii zapasowej
pg_dump -U postgres -d invoice_db > backup_$(date +%Y%m%d).sql

# Przywracanie z kopii zapasowej
psql -U postgres -d invoice_db < backup_20230622.sql
```

### Modele AI

Zapisz wersjonowane kopie modeli w systemie plików lub repozytorium modeli.

## Rozwiązywanie problemów

### Sprawdź status kontenerów

```bash
docker-compose -f docker-compose.prod.yml ps
```

### Sprawdź logi

```bash
docker-compose -f docker-compose.prod.yml logs -f
```

### Testuj endpointy API

```bash
curl -X GET "http://localhost:8000/health"
curl -X GET "http://localhost:8000/status"
```

## Dodatkowe zasoby

- [Dokumentacja FastAPI](https://fastapi.tiangolo.com/)
- [Dokumentacja Docker](https://docs.docker.com/)
- [Dokumentacja Kubernetes](https://kubernetes.io/docs/home/)
- [Best practices dla aplikacji produkcyjnych](https://12factor.net/)
