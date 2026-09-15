import io
import logging
import time
from typing import Any, Callable

import chess
import chess.engine
import chess.pgn

from chess_guard.analyzer.cache import AnalysisCache
from chess_guard.config import (
    STOCKFISH_HASH_MB,
    STOCKFISH_MULTIPV,
    STOCKFISH_PATH,
    STOCKFISH_THREADS,
)

logger = logging.getLogger(__name__)


ANALYSIS_NODES = 5_000_000
ENGINE_VERSION = "sf17"


class StockfishAnalyzer: #нализ партии

    def __init__(self, cache: AnalysisCache | None = None):
        self.cache = cache or AnalysisCache()
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        self.engine.configure({
            "Threads": STOCKFISH_THREADS,
            "Hash": STOCKFISH_HASH_MB,
        })

        # Получение версии движка
        self.engine_id = self.engine.id.get("name", ENGINE_VERSION)
        logger.info(" Stockfish запущен: %s", self.engine_id)

    def analyze_position(
        self, board: chess.Board, multipv: int = STOCKFISH_MULTIPV
    ) -> list[dict[str, Any]]:
        #Анализ позиции с кэшированием по EPD
        epd = board.epd()  # FEN без счётчиков

        # Проверяем кэш
        cached = self.cache.get(epd, self.engine_id, multipv, ANALYSIS_NODES)
        if cached is not None:
            return cached

        # Запрос к движку
        info = self.engine.analyse(
            board,
            chess.engine.Limit(nodes=ANALYSIS_NODES),
            multipv=multipv,
            game=object(),  #сброс хеша для изоляции
        )

        # Преобразуем в сериализуемый формат
        result = []
        for pv_info in info:
            score = pv_info["score"].relative
            result.append({
                "move": pv_info["pv"][0].uci() if pv_info.get("pv") else None,
                "score_cp": score.score(mate_score=10000),
                "depth": pv_info.get("depth", 0),
            })

        # Сохраняем в кэш
        self.cache.set(epd, self.engine_id, multipv, ANALYSIS_NODES, result)
        return result

    def analyze_game(
        self,
        pgn_text: str,
        max_moves: int | None = None,
        progress_callback: Callable[[int, int, dict], None] | None = None,
    ) -> list[dict]:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            raise ValueError("Не удалось распарсить PGN")

        moves = list(game.mainline_moves())
        if not moves:
            raise ValueError("PGN не содержит ходов")

        total_moves = len(moves) if max_moves is None else min(len(moves), max_moves)

        results = []
        board = game.board()
        move_times: list[float] = []

        for move_number, move in enumerate(moves):
            if move_number < 8:
                board.push(move)
                continue

            if max_moves and move_number >= max_moves:
                break

            move_start = time.monotonic()

            # Анализ позиции ДО хода (топ-5 вариантов)
            top_lines = self.analyze_position(board)
            if not top_lines:
                board.push(move)
                if progress_callback:
                    progress_callback(move_number + 1, total_moves, {
                        "actual_move": move.uci(),
                        "skipped": True,
                        "elapsed_s": time.monotonic() - move_start,
                    })
                continue

            actual_move_uci = move.uci()
            top1_move = top_lines[0]["move"]
            top1_cp = top_lines[0]["score_cp"]

            # Ищем оценку сыгранного хода среди топ-5
            actual_cp = None
            for line in top_lines:
                if line.get("move") == actual_move_uci:
                    actual_cp = line["score_cp"]
                    break

            # Если хода нет в топ-5 — анализируем позицию ПОСЛЕ хода
            if actual_cp is None:
                board.push(move)
                after_lines = self.analyze_position(board)
                board.pop()
                if after_lines:
                    # Инвертируем знак (теперь ход соперника)
                    actual_cp = -after_lines[0]["score_cp"]
                else:
                    actual_cp = top1_cp - 100  # грубый штраф

            # Потеря в сантипешках (всегда >= 0)
            cp_loss = max(0, int(top1_cp) - int(actual_cp))

            # Нормализованная точность (как на lichess)
            # 0 cp_loss = 100% точность, 100 cp_loss ≈ 50%, 300+ ≈ 0%
            accuracy = max(0.0, min(1.0, 1.0 - cp_loss / 300.0))

            is_top1 = actual_move_uci == top1_move

            # Топ-N ходов (для признака match_top3 в ML-модели)
            top_moves_list = [line.get("move") for line in top_lines if line.get("move")]
            is_top3 = actual_move_uci in top_moves_list[:3]

            results.append({
                "ply": move_number + 1,
                "fen": board.fen(),
                "actual_move": actual_move_uci,
                "top1_move": top1_move,
                "top1_score_cp": top1_cp,
                "actual_score_cp": actual_cp,
                "cp_loss": cp_loss,
                "accuracy": accuracy,
                "is_top1": is_top1,
                "is_top3": is_top3,
                "top_moves": top_moves_list,
                "num_legal_moves": board.legal_moves.count(),
            })

            board.push(move)

            elapsed_s = time.monotonic() - move_start
            move_times.append(elapsed_s)

            if progress_callback:
                avg_s = sum(move_times) / len(move_times)
                remaining = total_moves - (move_number + 1)
                progress_callback(move_number + 1, total_moves, {
                    "actual_move": actual_move_uci,
                    "cp_loss": cp_loss,
                    "elapsed_s": elapsed_s,
                    "avg_s_per_move": avg_s,
                    "eta_s": max(0.0, avg_s * remaining),
                })

        return results

    def close(self):
        self.engine.quit()
        self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()