# Przewodnik rozwiązywania problemów

Ten przewodnik zawiera rozwiązania typowych problemów, z którymi możesz się spotkać podczas korzystania z systemu do przetwarzania faktur.

## Problemy ze środowiskiem i konfiguracją

### Problem z ładowaniem modelu

**Objawy**: Aplikacja nie uruchamia się, wyświetlając błędy związane z ładowaniem modelu lub "None is not a valid model".

**Rozwiązanie**:
1. Sprawdź, czy plik `.env` istnieje i zawiera poprawny `MODEL_NAME`:
   ```bash
   cat .env | grep MODEL_NAME
   ```
2. Upewnij się, że nazwa modelu jest poprawnym identyfikatorem modelu z Hugging Face (np. `facebook/opt-125m`)
3. Dla modeli prywatnych upewnij się, że jesteś uwierzytelniony:
   ```bash
   huggingface-cli login
   ```
4. Sprawdź logi aplikacji pod kątem szczegółowych informacji o błędach

### Zmienne środowiskowe nie są ładowane

**Objawy**: Zmiany w pliku `.env` nie są uwzględniane.

**Rozwiązanie**:
1. Upewnij się, że plik `.env` znajduje się w katalogu głównym projektu
2. Sprawdź, czy nazwy zmiennych są zapisane WIELKIMI_LITERAMI
3. Uruchom ponownie aplikację po wprowadzeniu zmian
4. Sprawdź literówki w nazwach zmiennych
5. Upewnij się, że nie ma błędów składniowych w pliku `.env`

### Problemy z CUDA/GPU

**Objawy**: Błędy związane z CUDA lub komunikaty o przejściu na tryb CPU.

**Rozwiązanie**:
1. Sprawdź, czy CUDA jest zainstalowane:
   ```bash
   nvidia-smi  # Powinien wyświetlić informacje o karcie graficznej
   nvcc --version  # Powinien wyświetlić wersję CUDA
   ```
2. Sprawdź zgodność PyTorch z CUDA:
   ```python
   import torch
   print(torch.cuda.is_available())  # Powinno zwrócić True
   print(torch.version.cuda)  # Powinno pasować do wersji CUDA
   ```
3. Spróbuj ustawić `CUDA_VISIBLE_DEVICES` w pliku `.env`:
   ```env
   CUDA_VISIBLE_DEVICES=0
   ```
4. Jeśli używasz Dockera, upewnij się, że kontener ma dostęp do GPU:
   ```bash
   docker run --gpus all ...
   ```

## Problemy z API

### Port jest już w użyciu

**Objawy**: Błąd informujący, że port 8088 (lub inny skonfigurowany port) jest już używany.

**Rozwiązanie**:
1. Zmień port w pliku `.env`:
   ```env
   PORT=8080
   ```
2. Lub znajdź i zakończ proces używający portu:
   ```bash
   # Linux/macOS
   sudo lsof -i :8088
   kill -9 <PID>
   
   # Windows
   netstat -ano | findstr :8088
   taskkill /PID <PID> /F
   ```
3. Sprawdź, czy inna instancja aplikacji nie jest już uruchomiona

### Błędy CORS

**Objawy**: W konsoli przeglądarki pojawiają się błędy CORS przy wykonywaniu żądań do API.

**Rozwiązanie**:
1. Upewnij się, że adres URL frontendu znajduje się na liście `CORS_ORIGINS` w pliku `.env`:
   ```env
   CORS_ORIGINS=http://localhost:3000,http://localhost:8080
   ```
2. W środowisku deweloperskim możesz zezwolić na wszystkie źródła (niezalecane w produkcji):
   ```env
   CORS_ORIGINS=*
   ```
3. Sprawdź, czy nagłówki CORS są prawidłowo ustawione w odpowiedziach serwera

## Problemy z wydajnością

### Wolne działanie aplikacji

**Objawy**: Aplikacja działa zbyt wolno.

**Rozwiązanie**:
1. Sprawdź wykorzystanie GPU:
   ```bash
   watch -n 1 nvidia-smi  # Monitoruj wykorzystanie GPU co sekundę
   ```
2. Spróbuj użyć mniejszego modelu lub włącz kwantyzację w pliku `.env`:
   ```env
   QUANTIZATION=awq  # lub nf4 dla nowszych kart
   ```
3. Zwiększ rozmiar wsadu (batch size) przy przetwarzaniu wielu dokumentów:
   ```env
   BATCH_SIZE=8
   ```
4. Sprawdź, czy aplikacja korzysta z GPU zamiast CPU
5. Zoptymalizuj rozmiar obrazów przed przetwarzaniem OCR

### Wysokie zużycie pamięci

**Objawy**: Aplikacja ulega awarii z powodu braku pamięci.

**Rozwiązanie**:
1. Zmniejsz wykorzystanie pamięci GPU w pliku `.env`:
   ```env
   GPU_MEMORY_UTILIZATION=0.8
   ```
2. Użyj mniejszego modelu
3. Włącz sprawdzanie gradientu (gradient checkpointing) jeśli trenujesz model:
   ```env
   GRADIENT_CHECKPOINTING=True
   ```
4. Ogranicz rozmiar przetwarzanych danych jednorazowo
5. Sprawdź wycieki pamięci w kodzie

## Debugowanie

### Włączanie szczegółowych logów

Ustaw poziom logowania w pliku `.env`:
```env
LOG_LEVEL=DEBUG
```

### Sprawdzanie logów

Domyślnie logi są zapisywane w pliku `logs/app.log`. Sprawdź je w poszukiwaniu szczegółowych komunikatów o błędach.

### Najczęstsze komunikaty błędów

#### "No module named 'transformers'"

Zainstaluj wymagane pakiety:
```bash
pip install -r requirements.txt
# lub
poetry install
```

#### "CUDA out of memory"

1. Zmniejsz rozmiar wsadu (batch size)
2. Użyj mniejszego modelu
3. Włącz zarządzanie pamięcią CUDA w pliku `.env`:
   ```env
   GPU_MEMORY_UTILIZATION=0.8
   ```
4. Spróbuj włączyć kwantyzację:
   ```env
   QUANTIZATION=awq
   ```

#### "ConnectionError: Couldn't reach server"

1. Sprawdź połączenie z internetem
2. Upewnij się, że masz dostęp do Hugging Face Hub
3. Sprawdź ustawienia proxy, jeśli korzystasz z niego w sieci firmowej

#### "Error loading shared libraries: libcuda.so.1"

1. Upewnij się, że sterowniki NVIDIA są poprawnie zainstalowane
2. Sprawdź, czy ścieżka do bibliotek CUDA jest dodana do zmiennej środowiskowej `LD_LIBRARY_PATH`

## Uzyskiwanie pomocy

Jeśli wypróbowałeś powyższe rozwiązania i nadal występują problemy:

1. Sprawdź [zgłoszenia na GitHubie](https://github.com/finofficer/pocro/issues) pod kątem podobnych problemów
2. Przy zgłaszaniu błędu uwzględnij:
   - Zawartość pliku `.env` (z usuniętymi danymi wrażliwymi)
   - Pełny komunikat błędu i stos wywołań (stack trace)
   - Kroki prowadzące do odtworzenia problemu
   - Informacje o środowisku (system operacyjny, wersja Pythona, wersja CUDA jeśli używane jest GPU)
   - Logi aplikacji

## Przydatne komendy

### Sprawdzanie wersji Pythona i pakietów
```bash
python --version
pip list | grep -E "torch|transformers|fastapi"
```

### Czyszczenie pamięci podręcznej PyTorch
```bash
# Usuwa załadowane modele z pamięci podręcznej
torch.cuda.empty_cache()
```

### Sprawdzanie wykorzystania dysku przez modele
```bash
du -sh ~/.cache/huggingface/hub/
du -sh ~/.cache/torch/
```

### Resetowanie środowiska
```bash
# Usuń środowisko wirtualne i zainstaluj ponownie
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Aktualizacja zależności
```bash
# Zaktualizuj wszystkie zależności do najnowszych wersji
poetry update
# lub
pip install --upgrade -r requirements.txt
```

## Wsparcie

W razie pytań lub problemów, skontaktuj się z zespołem wsparcia lub utwórz nowe zgłoszenie w repozytorium projektu.