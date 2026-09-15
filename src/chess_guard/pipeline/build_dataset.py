from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from chess_guard.config import DATA_ANALYZED, DATA_PROCESSED

SEED = 42
CHEAT_P = 0.9
SKIP_IF_NO_TOP = True


def load_analyzed_games(path: Path) -> list[dict]:
    games = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            games.append(json.loads(line))
    return games


def score_of_move(top_lines: list[dict], move_uci: str) -> int | None:
    for line in top_lines:
        if line.get("move") == move_uci:
            return line.get("score_cp")
    return None


def features_for_move(move: dict) -> dict | None:
    actual = move.get("actual_move")
    top1 = move.get("top1_move")
    if not actual or not top1:
        return None

    top_lines = move.get("top_lines") or [{"move": top1}]
    top_moves = [t.get("move") for t in top_lines if t.get("move")]

    cp_loss = int(move.get("cp_loss", 0))
    accuracy = float(move.get("accuracy", max(0.0, 1.0 - cp_loss / 300.0)))

    return {
        "match_top1": int(actual == top1),
        "match_top3": int(actual in top_moves[:3]),
        "cp_loss": cp_loss,
        "accuracy": accuracy,
    }


def expand_game_rows(game: dict, cheat: bool, rng: random.Random) -> list[dict]:
    game_id = game.get("game_id", "unknown")
    suffix = "_cheat" if cheat else "_clean"
    rows = []

    for move in game.get("moves", []):
        base = features_for_move(move)
        if base is None:
            continue

        actual = move["actual_move"]
        top1 = move.get("top1_move")
        replaced = False

        if cheat and top1 and rng.random() < CHEAT_P:
            actual = top1
            replaced = True
            base = {"match_top1": 1, "match_top3": 1, "cp_loss": 0, "accuracy": 1.0}

        rows.append({
            "game_id": f"{game_id}{suffix}",
            "source_game_id": game_id,
            "ply": move.get("ply"),
            "actual_move": actual,
            "replaced": int(replaced),
            "label_move": int(replaced) if cheat else 0,
            "label_game": int(cheat),
            **base,
        })

    return rows


def build_dataset() -> Path:
    analyzed_path = DATA_ANALYZED / "analyzed_games.jsonl"
    if not analyzed_path.exists():
        raise FileNotFoundError(
            f"Нет файла {analyzed_path}. Сначала дождитесь batch_analyze."
        )

    print(f"{analyzed_path} ...")
    games = load_analyzed_games(analyzed_path)
    print(f"   партий: {len(games)}")

    rng = random.Random(SEED)
    all_rows: list[dict] = []

    for g in games:
        # чистая копия
        all_rows.extend(expand_game_rows(g, cheat=False, rng=rng))
        # синтетический читер (p=CHEAT_P подмен)
        all_rows.extend(expand_game_rows(g, cheat=True, rng=rng))

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError(
            "Датасет пуст. Проверьте формат analyzed_games.jsonl "
            "(нужны actual_move, top1_move, top1_score_cp)."
        )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "moves.parquet"
    df.to_parquet(out, index=False)

    # --- sanity ---
    n_games = df["game_id"].nunique()
    n_clean = df.loc[df["label_game"] == 0, "game_id"].nunique()
    n_cheat = df.loc[df["label_game"] == 1, "game_id"].nunique()

    print("\n Датасет собран")
    print(f"   файл:     {out}")
    print(f"   строк:    {len(df)} (ходов)")
    print(f"   партий:   {n_games} (clean={n_clean}, cheat={n_cheat})")
    print("\n  Средние признаки:")
    print(
        df.groupby("label_game")[["match_top1", "match_top3", "cp_loss", "accuracy"]].mean()
        .mean()
        .round(3)
        .to_string()
    )
    print("\n label_move value_counts:")
    print(df["label_move"].value_counts().to_string())

    return out


if __name__ == "__main__":
    build_dataset()