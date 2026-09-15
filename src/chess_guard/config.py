import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_ANALYZED = PROJECT_ROOT / "data" / "analyzed"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BIN_DIR = PROJECT_ROOT / "bin"

STOCKFISH_PATH = "C:/Users/portn/PycharmProjects/PythonProject3/bin/stockfish.exe" #путь до stockfish
STOCKFISH_THREADS = 2
STOCKFISH_HASH_MB = 128
STOCKFISH_MOVETIME_MS = 100
STOCKFISH_MULTIPV = 5

# Lichess API
LICHESS_TOKEN = ""
LICHESS_RATE_LIMIT_DELAY = 1.5