#Клиент для выгрузки партий с Lichess API

import json
import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import berserk
from berserk.exceptions import ApiError, ResponseError

from chess_guard.config import LICHESS_RATE_LIMIT_DELAY, LICHESS_TOKEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LichessCollector:
    #Обёртка над berserk.Client с защитой от rate-limit и ретраями

    def __init__(self, token: str | None = None, delay: float = LICHESS_RATE_LIMIT_DELAY):
        self.token = token or LICHESS_TOKEN
        self.delay = delay

        session = berserk.TokenSession(self.token) if self.token else None
        self.client = berserk.Client(session=session)

    def fetch_user_games(
        self,
        username: str,
        max_games: int = 50,
        perf_types: list[str] | None = None,
        rated_only: bool = True,
        max_retries: int = 3,
    ) -> Generator[dict[str, Any], None, None]:
        if perf_types is None:
            perf_types = ["blitz", "rapid"]

        perf_str = ",".join(perf_types)
        retries = 0
        backoff = 10.0

        while retries <= max_retries:
            try:
                time.sleep(self.delay)

                games_stream = self.client.games.export_by_player(
                    username,
                    max=max_games,
                    rated=rated_only,
                    perf_type=perf_str,
                    clocks=True,
                    opening=True,
                    moves=True,
                    evals=False,
                )

                count = 0
                for game in games_stream:
                    if isinstance(game, str):
                        logger.warning(
                            "Получена PGN-строка вместо JSON. "
                        )
                        continue

                    # Валидация: проверяем наличие ходов и часов
                    if not game.get("moves") or "clocks" not in game:
                        continue

                    # Проверяем, что партия закончилась нормально
                    if len(game.get("moves", "").split()) < 10:
                        continue

                    yield game
                    count += 1

                logger.info("Выгружено %d партий пользователя %s", count, username)
                return

            except ResponseError as e:
                # HTTP 429: Too Many Requests
                if e.status_code == 429:
                    retries += 1
                    logger.warning(
                        "⚠️ 429 Too Many Requests от Lichess. Ждем %.1f сек (попытка %d/%d)...",
                        backoff,
                        retries,
                        max_retries,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error("Ошибка HTTP от Lichess API: %s", e)
                    break

            except ApiError as e:
                logger.error("API Error для пользователя %s: %s", username, e)
                break

            except TypeError as e:
                # Отлавливаем ошибку с as_json или другими несовместимыми параметрами
                logger.error(
                    "Несовместимость версии berserk: %s\n",
                    e,
                )
                break

            except Exception as e:
                retries += 1
                logger.warning("Сетевой сбой: %s. Повтор через %.1f сек...", e, backoff)
                time.sleep(backoff)
                backoff *= 2

        logger.error(
            "Не удалось скачать партии для %s после %d попыток.", username, max_retries
        )


import json
from datetime import datetime
from pathlib import Path
from typing import Any


def datetime_handler(obj):
    #Конвертер datetime объектов в ISO-строки для JSON
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_games_to_jsonl(games: list[dict[str, Any]], filepath: Path) -> int:
    #Сохраняет список словарей в формате JSONL (по объекту на строку)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    with open(filepath, "a", encoding="utf-8") as f:
        for game in games:
            # Добавляем обработчик datetime через параметр default
            line = json.dumps(game, ensure_ascii=False, default=datetime_handler)
            f.write(line + "\n")
            saved_count += 1

    return saved_count