#Скрипт для сборки обучающего датасета партий по рейтинговым группам

from pathlib import Path
from tqdm import tqdm

from chess_guard.collector.lichess_client import LichessCollector, save_games_to_jsonl
from chess_guard.config import DATA_RAW


TARGET_COHORTS = {
    "tier_1500_1800": [
        "taparillo",
        "BigFlyOhtani",
        "CarleaX25",
        "yashgawas2016",
        "jesuscal",
        "JimGon"
    ],
    "tier_1800_2100": [
        "MD1985",
        "Rubenmachado",
        "bogart66",
        "Astrian-Rocky",
        "Majofran",
        "richi911007",
    ],
    "tier_2100_plus": [
        "penguingim1",
        "nihalsarin2004",
        "sparsely23",
        "lovlas",
        "Solvictus",
        "nitsugan"
    ]
}


def collect_rating_cohort(
    cohort_name: str,
    seed_users: list[str],
    collector: LichessCollector,
    target_games: int = 120,
    output_dir: Path = DATA_RAW,
) -> Path:
    #Собирает нужное число партий для конкретной когорты и сохраняет
    jsonl_path = output_dir / f"{cohort_name}.jsonl"

    # Если файл уже есть, очистим его перед новым сбором
    if jsonl_path.exists():
        jsonl_path.unlink()

    collected_games: list[dict] = []
    collected_game_ids: set[str] = set()

    pbar = tqdm(total=target_games, desc=f"Сбор [{cohort_name}]", unit="партий")

    for user in seed_users:
        if len(collected_games) >= target_games:
            break

        # Запрашиваем по 30-40 партий у каждого игрока
        games_generator = collector.fetch_user_games(
            username=user,
            max_games=40,
            perf_types=["blitz", "rapid"],
            rated_only=True
        )

        batch = []
        for game in games_generator:
            game_id = game["id"]
            if game_id not in collected_game_ids:
                collected_game_ids.add(game_id)
                batch.append(game)
                pbar.update(1)

                if len(collected_games) + len(batch) >= target_games:
                    break

        collected_games.extend(batch)
        if batch:
            save_games_to_jsonl(batch, jsonl_path)

    pbar.close()
    return jsonl_path


def run_collection(games_per_cohort: int = 120):
    print("Старт сбора корпуса партий с Lichess API")
    collector = LichessCollector()

    for cohort_name, users in TARGET_COHORTS.items():
        out_file = collect_rating_cohort(
            cohort_name=cohort_name,
            seed_users=users,
            collector=collector,
            target_games=games_per_cohort,
        )
        print(f"Сохранено в: {out_file.resolve()}\n")

    print("Сбор данных полностью завершён!")


if __name__ == "__main__":
    run_collection(games_per_cohort=120)