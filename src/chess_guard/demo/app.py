import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from chess_guard.analyzer.engine import StockfishAnalyzer
from chess_guard.config import MODELS_DIR

FEATURE_COLS = ["match_top1", "match_top3", "cp_loss", "accuracy"]


@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "lgbm.joblib"
    meta_path = MODELS_DIR / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        st.error("Модель не найдена. Запустите train_and_score.")
        st.stop()
    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta


def compute_cheat_probability(df: pd.DataFrame, model) -> pd.Series:
    p_move = model.predict_proba(df[FEATURE_COLS])[:, 1]
    p_cheat = pd.Series(p_move, index=df.index).expanding().mean()
    return p_cheat


def analyze_pgn(pgn_text: str, model, analyze_color: str = "both", progress_callback=None):
    with StockfishAnalyzer() as analyzer:
        move_analysis = analyzer.analyze_game(
            pgn_text, max_moves=None, progress_callback=progress_callback
        )

    if not move_analysis:
        return None

    rows = []
    for m in move_analysis:
        if m.get("top1_move") is None:
            continue

        ply = m["ply"]

        # Фильтр по цвету
        # Нечётные ply = белые, чётные = чёрные
        if analyze_color == "white" and ply % 2 == 0:
            continue
        if analyze_color == "black" and ply % 2 == 1:
            continue

        rows.append({
            "ply": ply,
            "actual_move": m["actual_move"],
            "top1_move": m.get("top1_move", "—"),
            # те же 4 признака, что и в build_dataset.py / train_and_score.py
            "match_top1": int(m["is_top1"]),
            "match_top3": int(m.get("is_top3", m["is_top1"])),
            "cp_loss": int(m.get("cp_loss", 0)),
            "accuracy": float(m.get("accuracy", 1.0)),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["p_cheat"] = compute_cheat_probability(df, model)
    df["top1_rate"] = df["match_top1"].expanding().mean()
    return df


def get_verdict(df: pd.DataFrame, threshold: float) -> tuple[str, str]:
    final_p = df["p_cheat"].iloc[-1]
    warn_zone = threshold * 0.6

    if final_p >= threshold:
        return "CHEAT", "red"
    elif final_p >= warn_zone:
        return "ПОДОЗРИТЕЛЬНО", "orange"
    else:
        return "CLEAN", "green"


def main():
    st.set_page_config(
        page_title="Chess Guard", page_icon="♟️", layout="wide"
    )

    st.title("Chess Guard")
    st.markdown(
        "Вставьте PGN партии"
    )

    model, meta = load_model()

    # --- Sidebar ---
    st.sidebar.header("ℹМодель")
    test_m = meta.get("metrics", {}).get("test", {})
    st.sidebar.metric("AUC (test)", f"{test_m.get('auc_game', 0):.3f}")
    st.sidebar.metric("FPR (test)", f"{test_m.get('fpr', 0):.1%}")
    st.sidebar.metric("Recall", f"{test_m.get('recall', 0):.1%}")

    threshold = meta.get("threshold", 0.5)


    # --- PGN input ---
    default_pgn = """ """

    pgn_input = st.text_area(
        "Вставьте PGN:",
        value=default_pgn,
        height=200,
    )
    col_left, col_right = st.columns(2)
    with col_left:
        analyze_color = st.radio(
            "Анализировать ходы:",
            ["both", "white", "black"],
            format_func=lambda x: {
                "both": "Оба игрока",
                "white": "Только белые",
                "black": "Только чёрные",
            }[x],
            horizontal=True,
        )
    with col_right:
        st.info(
            "Если проверяете конкретного игрока — "
            "выберите его цвет"
        )

    if st.button("Анализировать", type="primary"):
        progress_bar = st.progress(0, text="Запуска Stockfish...")
        status_line = st.empty()

        def _update_progress(current: int, total: int, info: dict):
            pct = current / total if total else 0.0
            move = info.get("actual_move", "—")
            eta_s = info.get("eta_s")
            avg_s = info.get("avg_s_per_move")

            eta_txt = f", осталось ~{eta_s:.0f} сек" if eta_s is not None else ""
            progress_bar.progress(min(pct, 1.0), text=f"Ход {current}/{total} ({move}){eta_txt}")

            if avg_s is not None:
                status_line.caption(
                    f" ~{avg_s:.1f} сек/ход при текущих настройках "
                )

        df = analyze_pgn(
            pgn_input, model, analyze_color=analyze_color,
            progress_callback=_update_progress,
        )

        progress_bar.empty()
        status_line.empty()

        if df is None:
            st.error("Не удалось проанализировать партию.")
            st.stop()

        verdict, color = get_verdict(df, threshold)
        final_p = df["p_cheat"].iloc[-1]
        top1_rate = df["match_top1"].mean()

        st.success(f"Проанализировано ходов: {len(df)}")

        # --- Метрики ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Вердикт", verdict)
        with col2:
            st.metric(
                "Вероятность читерства",
                f"{final_p:.1%}",
                help="P(читерство), выданная обученной LightGBM-моделью "
                     "(накопительное среднее по ходам)"
            )
        with col3:
            delta_from_baseline = top1_rate - 0.43
            st.metric(
                " Доля топ-1 ходов",
                f"{top1_rate:.1%}",
                delta=f"{delta_from_baseline:+.1%} от нормы",
                delta_color="inverse",
            )

        # --- График ---
        st.subheader("Вероятность читерства по ходам")

        chart_data = pd.DataFrame({
            "P(cheat)": df["p_cheat"].values,
            "Доля топ-1": df["top1_rate"].values,
        }, index=df["ply"].values)

        st.line_chart(chart_data, width="stretch")

        st.caption(
            f"Порог модели (P ≥ {threshold:.2f}) подобран под FPR ≤ "
            f"{meta.get('target_fpr', 0.05):.0%} на train-выборке "
            f"(train_and_score.py). Доля топ-1 в этой партии: {top1_rate:.1%}."
        )

        # --- Подозрительные ходы ---
        # --- Подозрительные ходы ---
        st.subheader("Все топ-1 ходы")
        top1_moves = df[df["match_top1"] == 1][
            ["ply", "actual_move", "top1_move", "p_cheat"]
        ].copy()
        top1_moves["p_cheat"] = top1_moves["p_cheat"].map(lambda x: f"{x:.1%}")

        if not top1_moves.empty:
            st.dataframe(top1_moves, width="stretch")

        # --- Полная таблица ---
        with st.expander("Полная таблица"):
            display_df = df.copy()
            display_df["p_cheat"] = display_df["p_cheat"].map(lambda x: f"{x:.1%}")
            display_df["top1_rate"] = display_df["top1_rate"].map(lambda x: f"{x:.1%}")
            st.dataframe(display_df, width="stretch")


if __name__ == "__main__":
    main()