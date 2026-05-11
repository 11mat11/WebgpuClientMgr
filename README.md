# WebGPU Thesis Benchmark Client

Profesjonalny klient benchmarkujacy REST API WebGPU/CUDA opisane w `openapi.json`.

## Wymagania

- Python 3.14+ (lub 3.11+ z asyncio TaskGroup)
- Serwer REST uruchomiony pod adresem `SERVER_URL`

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

Pojedynczy pipeline:

```powershell
python main.py --mode single --target matrix
```

Pelny benchmark (sekwencyjny + stres):

```powershell
python main.py --mode full
```

Test stresowy (wspolbiezny):

```powershell
python main.py --mode stress
```

## Wyniki

Wyniki zapisuja sie do `results/` jako CSV o nazwie zbudowanej z danych `/gpu/info` oraz daty.

Kolumna `run_mode` rozroznia pomiary sekwencyjne i stresowe. Wykresy (backend/gpu RTT) laduja w tym samym katalogu.

Gdy `USE_CUDA=1`, testy uruchamiaja sie zarowno dla WebGPU, jak i CUDA, osobno dla kazdego wariantu `optimized`.
