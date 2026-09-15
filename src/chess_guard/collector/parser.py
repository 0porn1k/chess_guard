"""Парсер сырых JSONL партий и извлечение таймингов."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedMove:
    ply: int
    move_san: str
    clock_seconds_left: float
    time_spent: float


@dataclass
class ParsedGame:
    game_id: str
    white_player: str
    black_player: str
    white_rating: int
    black_rating: int
    speed: str
    moves: list[str]
    clocks: list[float]
    time_spent: list[float]


def parse_lichess_json(game_dict: dict) -> ParsedGame:
    moves = game_dict.get("moves", "").split()
    raw_clocks = game_dict.get("clocks", [])  # обычно в сантисекундах (1/100 сек) или секундах

    # Переводим часы в секунды
    # Если значения > 1000 — скорее всего это сотые доли секунды
    clocks_sec = []
    for c in raw_clocks:
        clocks_sec.append(c / 100.0 if c > 1000 else float(c))

    # Вычисляем ΔT (время раздумий на каждом ходу)
    time_spent = []
    # Первые ходы белых и черных отсчитываются от начального контроля времени
    for i in range(len(clocks_sec)):
        if i == 0 or i == 1:
            # Первый ход белых / первый ход черных
            time_spent.append(1.0)  # базовое значение
        else:
            prev_same_color_clock = clocks_sec[i - 2]
            spent = max(0.1, prev_same_color_clock - clocks_sec[i])
            time_spent.append(round(spent, 2))

    return ParsedGame(
        game_id=game_dict["id"],
        white_player=game_dict["players"]["white"].get("user", {}).get("name", "anon"),
        black_player=game_dict["players"]["black"].get("user", {}).get("name", "anon"),
        white_rating=game_dict["players"]["white"].get("rating", 1500),
        black_rating=game_dict["players"]["black"].get("rating", 1500),
        speed=game_dict.get("speed", "blitz"),
        moves=moves,
        clocks=clocks_sec,
        time_spent=time_spent,
    )


def read_jsonl(filepath: Path) -> list[ParsedGame]:
    games = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            game_data = json.loads(line)
            games.append(parse_lichess_json(game_data))
    return games