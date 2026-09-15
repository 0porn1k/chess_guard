#Генерация движковой партии

import chess
import chess.engine
import chess.pgn
from datetime import datetime

from chess_guard.config import STOCKFISH_PATH
from chess_guard.analyzer.engine import ANALYSIS_NODES


def generate_engine_vs_engine(moves: int = 40) -> str:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Engine vs Engine (cheater simulation)"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "Stockfish_cheater"
    game.headers["Black"] = "Stockfish_cheater"
    node = game

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        engine.configure({"Threads": 1, "Hash": 128})

        for _ in range(moves * 2):
            if board.is_game_over():
                break

            result = engine.play(
                board,
                chess.engine.Limit(nodes=ANALYSIS_NODES),
                game=object(),
            )
            node = node.add_variation(result.move)
            board.push(result.move)

    game.headers["Result"] = board.result() if board.is_game_over() else "*"
    return str(game)


if __name__ == "__main__":
    print(f"Генерируем партию (nodes={ANALYSIS_NODES:,})...")
    pgn = generate_engine_vs_engine(moves=35)

    output = "data/synthetic_cheater.pgn"
    with open(output, "w", encoding="utf-8") as f:
        f.write(pgn)

    print(f"Сохранено: {output}")
    print(pgn)