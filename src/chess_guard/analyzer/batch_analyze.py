#Пакетный анализ всех партий

import json
import logging
import multiprocessing as mp

from tqdm import tqdm

from chess_guard.analyzer.cache import AnalysisCache
from chess_guard.analyzer.engine import StockfishAnalyzer
from chess_guard.config import DATA_ANALYZED, DATA_RAW

logging.basicConfig(level=logging.WARNING)  # Меньше логов в параллельном режиме
logger = logging.getLogger(__name__)


def analyze_single_game(game_data: dict) -> dict | None:
    try:
        # Каждый процесс создаёт свой движок и подключается к общей БД
        with StockfishAnalyzer() as analyzer:
            moves_str = game_data.get("moves", "")

            # Минимальный PGN для парсера
            pgn = f'[Event "?"]\n[Site "?"]\n\n{moves_str}'

            move_analysis = analyzer.analyze_game(pgn)

            return {
                "game_id": game_data["id"],
                "white": game_data["players"]["white"].get("user", {}).get("name", "?"),
                "black": game_data["players"]["black"].get("user", {}).get("name", "?"),
                "moves": move_analysis,
                "clocks": game_data.get("clocks", []),
            }
    except Exception as e:
        logger.error("Ошибка при анализе партии %s: %s", game_data.get("id", "?"), e)
        return None


def load_all_games() -> list[dict]:
    games = []
    for jsonl_file in DATA_RAW.glob("*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    games.append(json.loads(line))
    return games


def run_batch_analysis(num_processes: int | None = None):
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)  # Оставляем 1 ядро свободным

    print(f"Загрузка партий из {DATA_RAW}...")
    games = load_all_games()
    print(f"Найдено партий: {len(games)}")

    if not games:
        print("Нет партий для анализа")
        return

    print(f"Запуск анализа")

    # Используем Pool для распараллеливания
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(analyze_single_game, games),
                total=len(games),
                desc="Анализ партий",
                unit="партия",
            )
        )

    # Фильтруем None (ошибки анализа)
    results = [r for r in results if r is not None]

    # Сохраняем результаты
    output_file = DATA_ANALYZED / "analyzed_games.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Статистика кэша
    cache = AnalysisCache()
    stats = cache.stats()
    cache.close()

    print(f"\n Анализ завершён")
    print(f"   Проанализировано партий: {len(results)}")
    print(f"   Сохранено в: {output_file}")
    print(f"   Кэшировано позиций: {stats['cached_positions']}")


if __name__ == "__main__":
    run_batch_analysis()