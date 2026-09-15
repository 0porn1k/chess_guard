"""День 5: LightGBM + простая агрегация + порог FPR."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from chess_guard.config import DATA_PROCESSED, MODELS_DIR

FEATURE_COLS = ["match_top1", "match_top3", "cp_loss", "accuracy"]
SEED = 42
TARGET_FPR = 0.05


class GameScorer:
    def __init__(self, model, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.history: list[float] = []

    def update(self, features: dict | pd.DataFrame) -> float:
        if isinstance(features, dict):
            x = pd.DataFrame([features], columns=FEATURE_COLS)
        else:
            x = features

        p_move = float(self.model.predict_proba(x)[0, 1])
        self.history.append(p_move)
        p_game = float(np.mean(self.history))  # просто среднее
        return p_game

    def verdict(self, p: float | None = None) -> str:
        if p is None:
            p = self.history[-1] if self.history else 0.5
        return "CHEAT" if p >= self.threshold else "CLEAN"


def load_df() -> pd.DataFrame:
    path = DATA_PROCESSED / "moves.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}. Сначала build_dataset.")
    return pd.read_parquet(path)


def split_by_game(df: pd.DataFrame, test_size: float = 0.25):
    groups = df["game_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def train_model(train_df: pd.DataFrame) -> LGBMClassifier:
    X = train_df[FEATURE_COLS]
    y = train_df["label_move"]

    model = LGBMClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def move_level_auc(model, df: pd.DataFrame) -> float:
    proba = model.predict_proba(df[FEATURE_COLS])[:, 1]
    return float(roc_auc_score(df["label_move"], proba))


def score_games(model, df: pd.DataFrame) -> pd.DataFrame:

    rows = []
    for game_id, g in df.groupby("game_id", sort=False):
        g = g.sort_values("ply")
        X = g[FEATURE_COLS]
        p_moves = model.predict_proba(X)[:, 1]
        p_game = float(np.mean(p_moves))

        rows.append({
            "game_id": game_id,
            "label_game": int(g["label_game"].iloc[0]),
            "p_game": p_game,
            "n_moves": len(g),
        })
    return pd.DataFrame(rows)


def pick_threshold(game_scores: pd.DataFrame, target_fpr: float = TARGET_FPR) -> float:
    y_true = game_scores["label_game"].values
    y_score = game_scores["p_game"].values

    fpr_c, tpr_c, thr_c = roc_curve(y_true, y_score)

    # берём максимальный порог, при котором FPR <= target
    valid = np.where(fpr_c <= target_fpr)[0]
    if len(valid) == 0:
        # fallback: Youden
        idx = int(np.argmax(tpr_c - fpr_c))
        return float(thr_c[idx]) if idx < len(thr_c) else 0.5

    idx = valid[-1]
    return float(thr_c[idx])


def eval_at_threshold(game_scores: pd.DataFrame, threshold: float) -> dict:
    y = game_scores["label_game"].values
    p = game_scores["p_game"].values
    pred = (p >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    fpr = fp / (fp + tn + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    acc = (tp + tn) / max(len(y), 1)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

    return {
        "auc_game": auc,
        "fpr": float(fpr),
        "recall": float(rec),
        "accuracy": float(acc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": float(threshold),
    }


def save_artifacts(model, threshold: float, metrics: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lgbm.joblib"
    meta_path = MODELS_DIR / "meta.json"

    joblib.dump(model, model_path)
    meta = {
        "feature_cols": FEATURE_COLS,
        "threshold": threshold,
        "target_fpr": TARGET_FPR,
        "metrics": metrics,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n Сохранено:")
    print(f"   {model_path}")
    print(f"   {meta_path}")


def run() -> None:
    print(" Загрузка moves.parquet ...")
    df = load_df()
    print(f"   строк={len(df)}, партий={df['game_id'].nunique()}")

    train_df, test_df = split_by_game(df, test_size=0.25)
    print(
        f"   train: {train_df['game_id'].nunique()} партий / {len(train_df)} ходов | "
        f"test: {test_df['game_id'].nunique()} партий / {len(test_df)} ходов"
    )

    print("\n Обучение LightGBM ...")
    model = train_model(train_df)

    auc_move_tr = move_level_auc(model, train_df)
    auc_move_te = move_level_auc(model, test_df)
    print(f"   AUC (ход) train={auc_move_tr:.3f}  test={auc_move_te:.3f}")

    print("\n Скоринг партий (среднее вероятностей) ...")
    train_games = score_games(model, train_df)
    test_games = score_games(model, test_df)

    threshold = pick_threshold(train_games, target_fpr=TARGET_FPR)
    metrics_train = eval_at_threshold(train_games, threshold)
    metrics_test = eval_at_threshold(test_games, threshold)

    print(f"\n threshold = {threshold:.4f} (target FPR ≤ {TARGET_FPR:.0%})")
    print("\n TRAIN (game-level):")
    print(
        f"   AUC={metrics_train['auc_game']:.3f}  "
        f"FPR={metrics_train['fpr']:.3f}  "
        f"Recall={metrics_train['recall']:.3f}  "
        f"Acc={metrics_train['accuracy']:.3f}  "
        f"FP={metrics_train['fp']} FN={metrics_train['fn']}"
    )
    print(" TEST (game-level):")
    print(
        f"   AUC={metrics_test['auc_game']:.3f}  "
        f"FPR={metrics_test['fpr']:.3f}  "
        f"Recall={metrics_test['recall']:.3f}  "
        f"Acc={metrics_test['accuracy']:.3f}  "
        f"FP={metrics_test['fp']} FN={metrics_test['fn']}"
    )

    save_artifacts(
        model,
        threshold,
        {"train": metrics_train, "test": metrics_test, "auc_move_test": auc_move_te},
    )

    print("\n Smoke GameScorer:")
    for label, name in [(1, "cheat"), (0, "clean")]:
        sample_id = test_games.loc[test_games["label_game"] == label, "game_id"]
        if sample_id.empty:
            continue
        gid = sample_id.iloc[0]
        sub = test_df[test_df["game_id"] == gid].sort_values("ply")
        scorer = GameScorer(model, threshold=threshold)
        p = 0.5
        for _, row in sub.iterrows():
            p = scorer.update(row[FEATURE_COLS].to_dict())
        print(f"   {name}: {gid[:40]:40s}  P={p:.3f}  → {scorer.verdict(p)}")



if __name__ == "__main__":
    run()