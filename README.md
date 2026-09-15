Полный репозиторий - https://github.com/0porn1k/chess_guard


Требования
Python 3.12
uv — менеджер зависимостей и окружений
Бинарник Stockfish 
Токен Lichess API (опционально) — https://lichess.org/account/oauth/token
---
Установка
```bash
# Установка uv 
curl -LsSf https://astral.sh/uv/install.sh | sh        # Linux / macOS
# irm https://astral.sh/uv/install.ps1 | iex            # Windows PowerShell


uv sync
```


Все последующие команды запускаются через `uv run ...`
Перед первым запуском откройте `config.py` и укажите:
`STOCKFISH_PATH` — путь до скачанного бинарника Stockfish (stockfish 17 https://stockfishchess.org/download/)
`LICHESS_TOKEN` — токен Lichess API(опцианально)


В проекте уже есть натренированная модель, можно обучить свою (пункт 1-4), или запуститиь уже существующую (пункт 5)

1. Сбор корпуса партий с Lichess
```bash
uv run python src/chess_guard/collector/collect_corpus.py
```
Собирает партии для трёх рейтинговых когорт (1500–1800, 1800–2100, 2100+) от
заданного набора seed-аккаунтов через `LichessCollector` и сохраняет их в
`data/raw/<когорта>.jsonl`.

2. Пакетный анализ партий движком
```bash
uv run python src/chess_guard/analyzer/batch_analyze.py
```
Читает все `data/raw/*.jsonl`, анализирует каждую партию через
`StockfishAnalyzer`, результат
сохраняет в `data/analyzed/analyzed_games.jsonl`. Повторный анализ уже
встречавшихся позиций не выполняется — используется SQLite-кэш
(`data/analyzed/stockfish_cache.db`).

3. Построение обучающего датасета
```bash
uv run python src/chess_guard/pipeline/build_dataset.py
```
Из `analyzed_games.jsonl` строит табличный датасет ходов
(`data/processed/moves.parquet`) с четырьмя признаками (`match_top1`,
`match_top3`, `cp_loss`, `accuracy`). Для каждой реальной партии
дополнительно генерируется её синтетическая "нарушительская" копия — часть
ходов подменяется на топ-1 вариант движка с вероятностью `CHEAT_P`.
4. Обучение модели
```bash
uv run python src/chess_guard/pipeline/train_and_score.py
```
Обучает классификатор LightGBM на `moves.parquet`, разбивая выборку на
train/test по `game_id`. Порог классификации подбирается по ROC-кривой под
целевой FPR ≤ 5 %. Результат — `models/lgbm.joblib` (сама модель) и
`models/meta.json` (признаки, порог, метрики качества).
5. Запуск демонстрационного интерфейса
```bash
uv run streamlit run src/chess_guard/demo/app.py    
```
Открывает веб-интерфейс, туда вставить партию (в конце файла есть готовые партии разного рейтинга)

```bash
uv run python src/chess_guard/analyzer/inspect_results.py        # быстрый просмотр analyzed_games.jsonl в консоли
uv run python src/chess_guard/model/generate_engine_game.py    # генерация полностью движковой партии для калибровки/тестов
```
---
Назначение файлов
Файл	Назначение
`config.py`	Центральная конфигурация: пути проекта, параметры Stockfish, токен Lichess
`analyzer/engine.py`	`StockfishAnalyzer` — обёртка над движком: анализ позиции (MultiPV), расчёт `cp_loss`/`accuracy` для целой партии
`analyzer/cache.py`	`AnalysisCache` — SQLite-кэш результатов анализа позиций по ключу EPD
`collector/lichess_client.py`	`LichessCollector` — получение партий пользователя через Lichess API
`inder.py`	Поиск дополнительных активных игроков нужного рейтинга (арены, Lichess TV)
`collect_corpus.py`	Оркестрация сбора корпуса партий по рейтинговым когортам → `data/raw/*.jsonl`
`batch_analyze.py`	Параллельный анализ всех собранных партий движком → `data/analyzed/analyzed_games.jsonl`
`build_dataset.py`	Построение признаков и синтетической обучающей выборки → `data/processed/moves.parquet`
`train_and_score.py`	Обучение LightGBM, подбор порога по ROC-кривой, сохранение модели → `models/`
`generate_engine_game.py`	Генерация партии "движок против движка" — эталонный пример полностью нечестной игры
`inspect_results.py`	Консольный просмотр содержимого `analyzed_games.jsonl`
`app.py`	Демонстрационный веб-интерфейс на Streamlit: загрузка PGN → анализ → вердикт

эксперементальные партии:
### 1000-1500
masterayati15(1118)
yashgawas2016(1649)
```
1. e4 c5 2. Nf3 Nc6 3. Bb5 g6 { B31 Sicilian Defense: Nyezhmetdinov-Rossolimo Attack, Fianchetto Variation } 4. d3 Bg7 5. c3 Nf6 6. Bg5 O-O 7. O-O a6 8. Bxc6 dxc6 9. Nbd2 Bg4 10. Bxf6 Bxf6 11. d4 cxd4 12. cxd4 Bxd4 13. Nb3 Bxb2 14. Rb1 Qxd1 15. Rfxd1 Bxf3 16. gxf3 Bf6 17. Nc5 b5 18. Nd7 Rfd8 19. Nxf6+ exf6 20. Rxd8+ Rxd8 21. f4 Rd4 22. Re1 c5 23. h4 c4 24. Re2 c3 25. Kf1 Rd2 26. Ke1 Rd4 27. e5 fxe5 28. fxe5 Kg7 29. e6 fxe6 30. Rxe6 c2 31. Re7+ Kf6 32. Rc7 Rd1+ 33. Ke2 c1=Q 34. Rxc1 Rxc1 35. Kd2 Rh1 36. Kc3 Rxh4 37. a3 Rc4+ 38. Kb3 h5 39. a4 h4 40. f4 h3 41. axb5 h2 42. Kxc4 axb5+ 43. Kxb5 h1=Q 44. Kc4 Qf3 45. Kd4 Qxf4+ 46. Kc5 g5 47. Kb6 g4 48. Kb7 g3 49. Ka8 g2 50. Ka7 g1=Q+ 51. Ka8 Qa1+ 52. Kb7 Qb4+ 53. Kc8 Qac3+ 54. Kd7 Qe7# { Black wins by checkmate. } 0-1
```
### 1500-2000
Astrian-Rock(1794)
Messi9090(1786)
```
1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 4. Nxd4 e5 { B32 Sicilian Defense: Löwenthal Variation } 5. Ne2 Nf6 6. Ng3 Bc5 7. Bd3 d6 8. h3 h6 9. a3 a5 10. O-O Be6 11. Nc3 g5 12. Qe2 Rg8 13. Be3 Bxe3 14. fxe3 Qe7 15. Nf5 Bxf5 16. Rxf5 g4 17. hxg4 Nxg4 18. Nd5 Qh4 19. Qxg4 Qxg4 20. Nf6+ Ke7 21. Nxg4 Rxg4 22. Raf1 Rf8 23. Be2 Rg7 24. Rf6 Rfg8 25. R6f2 Nd8 26. Bf3 Ne6 27. Rd1 Ng5 28. Kh2 h5 29. Rfd2 Nxf3+ 30. gxf3 Rg6 { Black wins on time. } 0-1
```

2000+
undercovernerd(2200)
chessmem(2700)
```
1. d4 Nf6 2. Bf4 { A45 Indian Defense: Accelerated London System } d5 3. e3 c5 4. Nc3 a6 5. dxc5 Nc6 6. a3 e5 7. Bg5 d4 8. Ne4?! { (0.54 → -0.09) Inaccuracy. Na4 was best. } (8. Na4 h6 9. Bxf6 Qxf6 10. Qf3 Be6 11. Bd3 O-O-O 12. e4 Qe7) 8... Be7?! { (-0.09 → 0.54) Inaccuracy. Qd5 was best. } (8... Qd5 9. Nxf6+ gxf6 10. Bxf6 Rg8 11. Nf3 Rg6 12. Bh4 Bxc5 13. c4 Qe6 14. b4) 9. Bxf6 gxf6 10. b4?! { (0.47 → -0.34) Inaccuracy. exd4 was best. } (10. exd4 f5 11. Nd6+ Bxd6 12. cxd6 exd4 13. Qd2 Qxd6 14. O-O-O Be6 15. Qg5 Qd5) 10... Bf5? { (-0.34 → 0.94) Mistake. f5 was best. } (10... f5 11. Nd2 Be6 12. Bc4 Qd7 13. exd4 Rg8 14. Ngf3 e4 15. Bxe6 Qxe6 16. O-O) 11. Ng3 Bg6 12. Nf3 dxe3 13. fxe3 Qxd1+ 14. Rxd1 Bxc2 15. Rd2 Bg6 16. Nh4 a5 17. b5? { (1.07 → -0.23) Mistake. Bb5 was best. } (17. Bb5 axb4 18. axb4 Ra1+ 19. Rd1 Rxd1+ 20. Kxd1 Kd7 21. Ngf5 Ra8 22. Kc2 Bf8) 17... Nd8 18. Rc2?? { (-0.30 → -6.38) Blunder. c6 was best. } (18. c6 bxc6 19. bxc6 Bxa3 20. Bb5 O-O 21. O-O Rc8 22. Rd5 Ne6 23. Rd7 Rc7) 18... Rc8?? { (-6.38 → -0.33) Blunder. Bxc2 was best. } (18... Bxc2 19. c6 Bxa3 20. Bc4 bxc6 21. bxc6 Ne6 22. O-O Bc5 23. Nhf5 Bxf5 24. Nxf5) 19. Nxg6?! { (-0.33 → -1.18) Inaccuracy. Ngf5 was best. } (19. Ngf5 Rxc5 20. Bc4 b6 21. O-O Rg8 22. h3 Ne6 23. Rcc1 Bxf5 24. Nxf5 a4) 19... hxg6 20. Ne4? { (-1.20 → -2.58) Mistake. c6 was best. } (20. c6 bxc6 21. bxc6 Bxa3 22. Bb5 Ne6 23. Ne4 Ke7 24. Ke2 f5 25. Nc3 Rhd8) 20... f5 21. Nd2 Rxc5 22. Rxc5 Bxc5 23. Nc4 a4 24. g3 e4 25. Be2 Ne6 26. Kf2 Ke7 27. Rd1? { (-2.70 → -5.18) Mistake. h4 was best. } (27. h4 f4 28. gxf4 Nxf4 29. Bd1 f5 30. Bxa4 g5 31. h5 Rxh5 32. Rxh5 Nxh5) 27... Rxh2+ 28. Ke1 Ng5 29. Rd5 b6 30. Kd1 Rg2?! { (-4.78 → -3.19) Inaccuracy. Rh1+ was best. } (30... Rh1+ 31. Kc2 Ne6 32. g4 fxg4 33. Bxg4 f5 34. Bd1 Kf6 35. Kd2 Rh2+ 36. Ke1) 31. Rxc5 bxc5 32. b6 Kd7?! { (-3.51 → -2.25) Inaccuracy. Rh2 was best. } (32... Rh2 33. Na5 Rh8 34. Nc6+ Kd6 35. Bb5 Nf3 36. b7 Kc7 37. b8=R Rxb8 38. Nxb8) 33. Ne5+ Kd8 34. Nc6+ Kc8 35. Ba6+ Kd7 36. b7? { (-2.03 → -4.20) Mistake. Ne5+ was best. } (36. Ne5+ Kd6 37. Kc1 Rh2 38. Bc8 Rh8 39. b7 Kc7 40. Nd7 Rxc8 41. bxc8=Q+ Kxc8) 36... Rb2 37. b8=Q Rxb8 38. Nxb8+ Kc7 39. Bb5 Kxb8 40. Bxa4 Kc7 41. Bb3?! { (-3.05 → -4.47) Inaccuracy. Be8 was best. } (41. Be8 Kd8 42. Bb5 Nf3 43. a4 Kc7 44. a5 Ne5 45. Kc2 Nc6 46. a6 Kb6) 41... Kb6 42. Kc2 f6 43. Kc3 Nf3 44. Bf7 g5?! { (-4.17 → -2.77) Inaccuracy. Ne5 was best. } (44... Ne5 45. Bb3 Ng4 46. Kd2 Kb5 47. Bd1 c4 48. Be2 Kc5 49. Bd1 c3+ 50. Kxc3) 45. Bg6 f4?? { (-1.93 → -0.12) Blunder. Nh2 was best. } (45... Nh2 46. Bxf5 Nf1 47. Bxe4 Nxg3 48. Bg6 f5 49. Kc4 g4 50. Kd3 Ka5 51. e4) 46. gxf4?! { (-0.12 → -0.73) Inaccuracy. exf4 was best. } (46. exf4 Nd4 47. fxg5 fxg5 48. Kd2 Ka5 49. Bxe4 Ka4 50. Kd3 Kxa3 51. Kc4 Nb3) 46... g4 47. Bxe4 Nh4 48. Kc4?? { (-0.69 → -5.01) Blunder. Kd2 was best. } (48. Kd2 g3 49. Ke2 c4 50. Bb1 Kc5 51. e4 g2) 48... g3 49. a4 g2 50. Bxg2 Nxg2 51. a5+ Kxa5 52. Kxc5 Nxe3 53. Kd6?! { (-6.80 → -81.15) Inaccuracy. Kd4 was best. } (53. Kd4 Nf5+ 54. Kd5 Ng7 55. Kc5 Ka4 56. Kd4 Kb5 57. Kd5 Kb4 58. Kd4 Nh5) 53... Kb5 54. Ke6?! { (-13.33 → Mate in 23) Checkmate is now unavoidable. f5 was best. } (54. f5 Kc4 55. Ke7 Nd5+ 56. Kd7 Kd4 57. Kc6 Ke5 58. Kb5 Nf4 59. Ka5 Kxf5) 54... f5 55. Ke5 Kc4 56. Kf6 Kd4 57. Kg5?! { (-13.99 → Mate in 17) Checkmate is now unavoidable. Ke7 was best. } (57. Ke7 Ng4 58. Kd6 Ke4 59. Kc5 Kxf4 60. Kd4 Kf3 61. Kd3 f4 62. Kc4 Kg3) 57... Ke4 { White resigns. } 0-1
```

### 1500 с подсказками

1500 игрок
Stockfish 5lvl chess.com
```
1. e4 d5 2. exd5 Qxd5 3. d4 Nc6 4. Nf3 Bg4 5. Be2 O-O-O 6. Nc3 Qa5 7. Be3 e5 8. Nxe5 Bxe2 9. Qxe2 Nxe5 10. dxe5 Ba3 11. Bd4 (11. Qg4+ Kb8 12. Qxg7 Bxb2 13. Rb1 Bxc3+ 14. Ke2 Ne7 15. Rhd1 Nd5 16. Qxf7 Bxe5 17. g3 Nc3+ 18. Kf3 Nxd1 19. Rxd1 Rxd1 20. Qe6 Bd4 21. Bxd4 Rxd4 22. c4 Rf8+ 23. Kg2 Qxa2 24. Qh6 Qxf2+ 25. Kh3 Qf1#)
```

engine 100_000 нод
```
1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O Nxe4 5. Re1 Nd6 6. Nxe5 Be7 7. Bf1 Nxe5 8. Rxe5 O-O 9. d4 Bf6 10. Re1 Re8 11. Bf4 Rxe1 12. Qxe1 Ne8 13. Qe3 d5 14. a4 Bf5 15. c3 a5 16. Nd2 h6 17. Nf3 Nd6 18. Qe2 Qe7 19. Re1 Qxe2 20. Bxe2 Rd8 21. h4 Bg4 22. Bg3 Kf8 23. Nd2 Bxe2 24. Rxe2 h5 25. Kf1 g6 26. Nb3 b6 27. Bxd6+ Rxd6 28. g3 c6 29. Ke1 Rd7 30. Nc1 Be7 31. Nd3 Bd6 32. Kd1 b5 33. b3 bxa4 34. bxa4 Rb7 35. Kc2 Re7 *
```