#SQLite-кэш результатов анализа движка для ускорения повторных запросов

import json
import sqlite3
from pathlib import Path
from typing import Any

from chess_guard.config import DATA_ANALYZED


class AnalysisCache:
    #Кэш анализа позиций

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = DATA_ANALYZED / "stockfish_cache.db"

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis (
                epd TEXT PRIMARY KEY,
                engine_version TEXT,
                multipv INTEGER,
                nodes INTEGER,
                result TEXT
            )
        """)
        self.conn.commit()

    def get(self, epd: str, engine_version: str, multipv: int, nodes: int) -> list[dict] | None:
        cursor = self.conn.execute(
            """
            SELECT result FROM analysis 
            WHERE epd = ? AND engine_version = ? AND multipv = ? AND nodes = ?
            """,
            (epd, engine_version, multipv, nodes),
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set(
        self, epd: str, engine_version: str, multipv: int, nodes: int, result: list[dict]
    ):
        self.conn.execute(
            """
            INSERT OR REPLACE INTO analysis (epd, engine_version, multipv, nodes, result)
            VALUES (?, ?, ?, ?, ?)
            """,
            (epd, engine_version, multipv, nodes, json.dumps(result)),
        )
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        #Статистика кэша
        cursor = self.conn.execute("SELECT COUNT(*) FROM analysis")
        count = cursor.fetchone()[0]
        return {"cached_positions": count}

    def close(self):
        self.conn.close()