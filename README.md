# WebGPU Thesis Benchmark Client

Profesjonalny klient benchmarkujacy REST API WebGPU/CUDA opisane w `openapi.json`.

## Wymagania

- Python 3.14+ (lub 3.11+ z asyncio TaskGroup)
- Serwer REST uruchomiony pod adresem `SERVER_URL`
- WebSocket w serwerze pod `/video/stream`

## Instalacja

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Konfiguracja

Ustaw zmienne w `.env` (na start skopiowano z `.env.example`).

```
SERVER_URL=http://127.0.0.1:3000
USE_CUDA=1
```

## Uruchomienie

Tryb szybki:

```powershell
python main.py --mode quick
```

Tryb szybki uruchamia tylko mniejsze rozmiary danych (szybka walidacja).

Tryb szybki z maksymalnymi rozmiarami:

```powershell
python main.py --mode quick --quick-max
```

Test wspolbiezny dla najwiekszej macierzy (1..N rownolegle):

```powershell
python main.py --mode full --matrix-concurrency-max 12
```

Pojedynczy pipeline:

```powershell
python main.py --mode single --target matrix
```

Pelny benchmark (sekwencyjny + stres):

```powershell
python main.py --mode full
```

Tryb pelny z testem obciazeniowym (duze requesty rownolegle):

```powershell
python main.py --mode full --load-test --load-concurrency 32 --load-requests 200
```

Test stresowy (wspolbiezny):

```powershell
python main.py --mode stress
```

## Wyniki

Wyniki zapisuja sie do `results/<nazwa>/` (nazwa z `/gpu/info` + data). Dla kazdego endpointu powstaje osobny katalog z:

- `tabelki*.csv` (osobno per endpoint/optimized/run_mode)
- wykresami `backend_duration_ms*.png`, `gpu_duration_ms*.png`, `client_rtt_ms*.png`
- dla testu wspolbieznych macierzy: `backend_duration_ms_bar*concurrency*.png` z +/-

Gdy `USE_CUDA=1`, testy uruchamiaja sie zarowno dla WebGPU, jak i CUDA, osobno dla kazdego wariantu `optimized`.
