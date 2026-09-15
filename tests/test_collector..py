import json
from pathlib import Path
from chess_guard.collector.lichess_client import save_games_to_jsonl
from chess_guard.collector.parser import parse_lichess_json


def test_save_and_parse_jsonl(tmp_path: Path):
    """Проверяем цикл сериализации и десериализации JSONL."""
    sample_game = {
        "id": "test1234",
        "rated": True,
        "speed": "blitz",
        "moves": "e4 e5 Nf3 Nc6 Bc4 Bc5",
        "clocks": [18000, 18000, 17800, 17900, 17500, 17600],
        "players": {
            "white": {"user": {"name": "Player1"}, "rating": 1600},
            "black": {"user": {"name": "Player2"}, "rating": 1650},
        },
    }

    test_file = tmp_path / "test_games.jsonl"
    count = save_games_to_jsonl([sample_game], test_file)
    assert count == 1
    assert test_file.exists()

    # Проверяем парсер
    with open(test_file, encoding="utf-8") as f:
        data = json.loads(f.readline())

    parsed = parse_lichess_json(data)
    assert parsed.game_id == "test1234"
    assert parsed.white_rating == 1600
    assert len(parsed.moves) == 6
    assert len(parsed.clocks) == 6
    assert len(parsed.time_spent) == 6