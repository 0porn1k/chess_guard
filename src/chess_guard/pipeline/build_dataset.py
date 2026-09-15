"""День 4: признаки + синтетические читеры → data/processed/moves.parquet"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from chess_guard.config import DATA_ANALYZED, DATA_PROCESSED

# --- настройки ---
SEED = 42
CHEAT_P = 0.5          # доля ходов, подменённых на top1 у «читера»
SKIP_IF_NO_TOP = True  # пропускать ходы без top1 от движка


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
    """Оценка сыгранного хода по multipv; None если хода нет в топ-N."""
    for line in top_lines:
        if line.get("move") == move_uci:
            return line.get("score_cp")
    return None


def features_for_move(move: dict) -> dict | None:
    """
    Из записи анализа хода собираем 3 признака.

    Ожидаемый формат move (из batch_analyze / engine):
      actual_move, is_top1, top1_move, top1_score_cp, ...
    Дополнительно, если есть полный multipv в move["top_lines"] — используем.
    """
    actual = move.get("actual_move")
    top1 = move.get("top1_move")
    top1_cp = move.get("top1_score_cp")

    if not actual or not top1 or top1_cp is None:
        if SKIP_IF_NO_TOP:
            return None
        return None

    # top_lines: если batch_analyze их ещё не пишет — восстанавливаем минимум
    top_lines = move.get("top_lines")
    if not top_lines:
        top_lines = [{
            "move": top1,
            "score_cp": top1_cp,
        }]

    top_moves = [t.get("move") for t in top_lines if t.get("move")]
    match_top1 = int(actual == top1)
    match_top3 = int(actual in top_moves[:3])

    actual_cp = score_of_move(top_lines, actual)
    if actual_cp is None:
        # ход не в multipv — грубая оценка: штраф (типично 50–150 cp)
        # для MVP фиксированный penalty, чтобы не звать Stockfish снова
        actual_cp = top1_cp - 80

    cp_loss = max(0, int(top1_cp) - int(actual_cp))

    return {
        "match_top1": match_top1,
        "match_top3": match_top3,
        "cp_loss": cp_loss,
    }


def expand_game_rows(game: dict, cheat: bool, rng: random.Random) -> list[dict]:
    """Одна партия → список строк (по одному ходу)."""
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
            # подмена хода на top1 движка
            actual = top1
            replaced = True
            base = {
                "match_top1": 1,
                "match_top3": 1,
                "cp_loss": 0,
            }

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

    print(f"📥 Читаю {analyzed_path} ...")
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

    print("\n✅ Датасет собран")
    print(f"   файл:     {out}")
    print(f"   строк:    {len(df)} (ходов)")
    print(f"   партий:   {n_games} (clean={n_clean}, cheat={n_cheat})")
    print("\n📊 Средние признаки:")
    print(
        df.groupby("label_game")[["match_top1", "match_top3", "cp_loss"]]
        .mean()
        .round(3)
        .to_string()
    )
    print("\n📊 label_move value_counts:")
    print(df["label_move"].value_counts().to_string())

    return out


if __name__ == "__main__":
    build_dataset()