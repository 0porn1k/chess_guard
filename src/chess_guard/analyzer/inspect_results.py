"""Просмотр результатов анализа."""

import json
from chess_guard.config import DATA_ANALYZED


def inspect_analyzed_games():
    filepath = DATA_ANALYZED / "analyzed_games.jsonl"

    if not filepath.exists():
        print("Файл analyzed_games.jsonl не найден!")
        return

    with open(filepath, encoding="utf-8") as f:
        games = [json.loads(line) for line in f if line.strip()]

    print(f"Всего проанализировано партий: {len(games)}\n")

    # Первая партия для примера
    sample = games[0]
    print(f"   Партия: {sample['white']} vs {sample['black']}")
    print(f"   ID: {sample['game_id']}")
    print(f"   Ходов проанализировано: {len(sample['moves'])}\n")

    # Первые 3 хода
    for move in sample["moves"][:3]:
        mark = "OK" if move["is_top1"] else "NO"
        print(
            f"   Ход {move['ply']:2d}: {move['actual_move']:5s} | "
            f"Топ-1: {move['top1_move']:5s} ({move['top1_score_cp']:+4d} cp) {mark}"
        )


if __name__ == "__main__":
    inspect_analyzed_games()