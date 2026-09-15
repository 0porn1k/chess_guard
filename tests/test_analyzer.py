from pathlib import Path
import chess.engine
from chess_guard.analyzer import analyze_game
from chess_guard.config import STOCKFISH_PATH


def test_stockfish_exists():
    path = Path(STOCKFISH_PATH)
    assert path.exists(), f"Файл Stockfish не найден: {STOCKFISH_PATH}"


def test_stockfish_engine_response():
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        assert "Stockfish" in engine.id["name"]


def test_analyze_game_output():
    pgn = "1. e4 e5 2. Nf3 Nc6"
    results = analyze_game(pgn, max_moves=2)
    assert len(results) == 2
    assert results[0]["actual_move"] == "e2e4"
    assert "top1_score_cp" in results[0]