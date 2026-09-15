"""День 6: Streamlit-демо — PGN → график вероятности читерства."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from chess_guard.analyzer.engine import StockfishAnalyzer
from chess_guard.config import MODELS_DIR

# ----------------------------------------------------------------------------
# Загрузка модели
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "lgbm.joblib"
    meta_path = MODELS_DIR / "meta.json"

    if not model_path.exists() or not meta_path.exists():
        st.error(
            f"❌ Модель не найдена в {MODELS_DIR}. "
            "Сначала запустите: `uv run python -m chess_guard.pipeline.train_and_score`"
        )
        st.stop()

    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta


# ----------------------------------------------------------------------------
# GameScorer (копия из train_and_score, упрощённая)
# ----------------------------------------------------------------------------
class GameScorer:
    def __init__(self, model, threshold: float = 0.5, feature_cols=None):
        self.model = model
        self.threshold = threshold
        self.feature_cols = feature_cols or ["match_top1", "match_top3", "cp_loss"]
        self.reset()

    def reset(self):
        self.history = []

    def update(self, features: dict) -> float:
        x = pd.DataFrame([features], columns=self.feature_cols)
        p_move = float(self.model.predict_proba(x)[0, 1])
        self.history.append(p_move)
        import numpy as np
        p_game = float(np.mean(self.history))
        return p_game

    def verdict(self, p: float | None = None) -> str:
        if p is None:
            p = self.history[-1] if self.history else 0.5
        return "🚨 CHEAT" if p >= self.threshold else "✅ CLEAN"


# ----------------------------------------------------------------------------
# Анализ партии
# ----------------------------------------------------------------------------
def analyze_pgn(pgn_text: str, model, meta):
    feature_cols = meta["feature_cols"]
    threshold = meta["threshold"]

    with StockfishAnalyzer() as analyzer:
        move_analysis = analyzer.analyze_game(pgn_text, max_moves=None)

    if not move_analysis:
        st.warning("⚠️ Партия пуста после пропуска дебюта (первые 8 ходов).")
        return None, None

    # Собираем признаки
    rows = []
    for m in move_analysis:
        top1_cp = m.get("top1_score_cp")
        if top1_cp is None:
            continue

        # упрощённая логика cp_loss (как в build_dataset)
        actual_cp = top1_cp if m["is_top1"] else top1_cp - 80
        cp_loss = max(0, int(top1_cp) - int(actual_cp))

        rows.append({
            "ply": m["ply"],
            "actual_move": m["actual_move"],
            "match_top1": int(m["is_top1"]),
            "match_top3": int(m["is_top1"]),  # нет полного multipv → дубль
            "cp_loss": cp_loss,
        })

    df = pd.DataFrame(rows)

    # Скоринг
    scorer = GameScorer(model, threshold=threshold, feature_cols=feature_cols)
    probs = []
    for _, row in df.iterrows():
        p = scorer.update(row[feature_cols].to_dict())
        probs.append(p)

    df["p_cheat"] = probs
    final_p = probs[-1] if probs else 0.5
    verdict = scorer.verdict(final_p)

    return df, verdict


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Chess Guard", page_icon="♟️", layout="wide")

    st.title("♟️ Chess Guard — детектор подсказок движка")
    st.markdown("Вставьте PGN партии → получите динамический график вероятности читерства.")

    model, meta = load_model()

    st.sidebar.header("ℹ️ О модели")
    st.sidebar.metric("AUC (test)", f"{meta['metrics']['test']['auc_game']:.3f}")
    st.sidebar.metric("FPR (test)", f"{meta['metrics']['test']['fpr']:.1%}")
    st.sidebar.metric("Recall (test)", f"{meta['metrics']['test']['recall']:.1%}")
    st.sidebar.metric("Порог", f"{meta['threshold']:.3f}")

    # Поле ввода PGN
    default_pgn = """[Event "Rated Blitz"]
[Site "https://lichess.org/xyz"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 
7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5 11. d4 Qc7 12. Nbd2 cxd4 
13. cxd4 Nc6 14. Nb3 a5 15. Be3 a4 16. Nbd2 Bd7 17. Rc1 Qb7 
18. d5 Na5 19. b4 axb3 20. axb3 Rfc8 21. Qe2 Qb6 22. Bd3 Rxc1 
23. Rxc1 Rc8 24. Rxc8+ Bxc8 25. Qc2 Bd7 26. Nf1 h6 27. N3d2 Nb7 
28. Nc4 Qc7 29. Qxc7 1-0"""

    pgn_input = st.text_area(
        "Вставьте PGN партии:",
        value=default_pgn,
        height=250,
    )

    if st.button("🔍 Анализировать", type="primary"):
        with st.spinner("⏳ Анализируем через Stockfish..."):
            df, verdict = analyze_pgn(pgn_input, model, meta)

        if df is None:
            st.stop()

        st.success(f"Анализ завершён! Проанализировано ходов: {len(df)}")

        # Вердикт
        final_p = df["p_cheat"].iloc[-1]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Вердикт", verdict)
        with col2:
            st.metric("📊 Финальная вероятность", f"{final_p:.1%}")

        # График
        st.subheader("📈 Динамика вероятности по ходам")
        chart_df = df[["ply", "p_cheat"]].set_index("ply")
        st.line_chart(chart_df, use_container_width=True)

        # Таблица подозрительных ходов
        st.subheader("🔎 Подозрительные ходы (P > 30%)")
        suspicious = df[df["p_cheat"] > 0.3][
            ["ply", "actual_move", "match_top1", "cp_loss", "p_cheat"]
        ]
        if not suspicious.empty:
            st.dataframe(suspicious, use_container_width=True)
        else:
            st.info("Нет ходов с высокой вероятностью читерства.")

        # Детальная таблица (свёрнуто)
        with st.expander("📋 Полная таблица анализа"):
            st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()