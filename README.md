# Instrukcja Uruchomienia Benchmarku (WebGPU vs CUDA)

## 🛠 Wymagania
- Python 3.11+
- Środowisko wirtualne (`.venv`)
- Zainstalowane zależności (`pip install -r requirements.txt`)
- Skonfigurowany plik `.env` (`SERVER_URL`, `USE_CUDA`)

---

## 🚀 Zalecana Procedura Badawcza

Aby zapewnić rzetelność wyników i stabilność sprzętu, należy postępować zgodnie z poniższymi krokami:

### KROK 1: Walidacja Limitów Sprzętowych (Quick-Max)
Zanim uruchomisz wielogodzinny test, sprawdź, czy Twoja karta graficzna i sterownik wytrzymają maksymalne obciążenie macierzami 12000x12000 i obrazami 8K.

- **Test z CUDA:**
  `python main.py --mode quick --quick-max`
- **Test bez CUDA (Samo WebGPU):**
  `python main.py --mode quick --quick-max --no-cuda`

> **Ważne:** Sprawdź, czy serwer nie wyrzucił błędu TDR (Device Hung) lub błędu braku pamięci (Out-Of-Memory). Jeśli test przeszedł pomyślnie, przejdź do KROKU 2. Jeśli nie, zapoznaj się z sekcją "Rozwiązywanie problemów sprzętowych" poniżej.

### KROK 2: Przygotowanie "Czystego" Środowiska
Aby uniknąć wpływu "śmieci" w pamięci na wyniki właściwe (problem zanieczyszczenia pamięci):

1. **Usuń tymczasowe dane:** Skasuj folder `results/` wygenerowany w Kroku 1.
2. **Restart serwera:** Koniecznie wyłącz i włącz ponownie serwer Node.js. To jedyny sposób na pełny reset kontekstu GPU i zrzucenie cache'u sterownika.

### KROK 3: Uruchomienie Pełnego Benchmarku (Full Mode)
Kiedy masz pewność, że sprzęt jest stabilny i środowisko jest czyste:

- **Benchmark z CUDA:**
  `python main.py --mode full`
- **Benchmark bez CUDA (Samo WebGPU):**
  `python main.py --mode full --no-cuda`

---

## ⚠️ Rozwiązywanie problemów sprzętowych (OOM / TDR)

Procedura badawcza zakłada testowanie sprzętu na granicach jego możliwości obliczeniowych oraz pojemności pamięci VRAM. W przypadku urządzeń o mniejszej ilości pamięci, test walidacyjny `quick-max` może zakończyć się awarią sterownika graficznego. W takiej sytuacji należy dostosować parametry obciążenia.

### Scenariusz A: Błąd w teście współbieżności (na samym końcu)
Jeśli wszystkie testy sekwencyjne przechodzą pomyślnie, ale serwer ulega awarii pod koniec działania (podczas równoległego wysyłania największych macierzy), oznacza to, że zsumowana wielkość alokacji dla wielu zadań przekracza fizyczny limit VRAM.

**Rozwiązanie:** Kod główny pozostaje bez zmian. Należy uruchomić pełny test, nadpisując domyślny limit współbieżności (5) za pomocą flagi `--matrix-concurrency-max` na wartość gwarantującą stabilność (np. 4):
`python main.py --mode full --matrix-concurrency-max 4`

### Scenariusz B: Błąd podczas testów sekwencyjnych
Jeśli awaria następuje już w trakcie testowania pojedynczych, największych elementów (np. macierzy 10000x10000 lub obrazów 8K), konieczna jest manualna redukcja maksymalnych rozmiarów danych w konfiguracji skryptu klienckiego.

**Rozwiązanie:** Należy wyedytować plik `main.py` (linie od 33 do 61) i zredukować lub usunąć największe wartości w głównych listach konfiguracyjnych:

```python
DEFAULT_ITERATIONS: Final[int] = 30
DEFAULT_STRESS_CONCURRENCY: Final[int] = 64
DEFAULT_STRESS_REQUESTS: Final[int] = 1000
DEFAULT_STRESS_XL_REQUESTS: Final[int] = 200
DEFAULT_LOAD_CONCURRENCY: Final[int] = 16
DEFAULT_LOAD_REQUESTS: Final[int] = 100
DEFAULT_MATRIX_CONCURRENCY_MAX: Final[int] = 5
MATRIX_SIZES: Final[list[int]] = [256, 500, 512, 1000, 1024, 2048, 3000, 4096, 5000, 8192, 10000]
MATRIX_SIZES_QUICK: Final[list[int]] = [256, 512]
IMAGE_SIZES: Final[list[tuple[int, int]]] = [
    (320, 180),    # 180p
    (640, 360),    # 360p
    (960, 540),    # 540p (qHD)
    (1280, 720),   # 720p (HD)
    (1600, 900),   # 900p (HD+)
    (1920, 1080),  # 1080p (Full HD)
    (2560, 1440),  # 1440p (QHD / 2.5K)
    (3840, 2160),  # 4K (UHD)
    (5120, 2880),  # 5K
    (6144, 3456),  # 6K
    (7680, 4320),  # 8K (UHD-2)
]
IMAGE_SIZES_QUICK: Final[list[tuple[int, int]]] = [(320, 180), (640, 360)]
VIDEO_FRAMES: Final[list[int]] = list(range(20))
VIDEO_FRAMES_QUICK: Final[list[int]] = [0, 1, 2]
VIDEO_QUALITIES: Final[list[str]] = ["1080p", "720p", "480p", "160p"]
VIDEO_QUALITIES_QUICK: Final[list[str]] = ["480p", "160p"]
RENDER_COUNTS: Final[list[int]] = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]
RENDER_COUNTS_QUICK: Final[list[int]] = [500, 1000]
```