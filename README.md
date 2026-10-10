# VSCodeWorkspace

Personal Python projects, one folder per area. Scripts import their siblings by name, so run them from inside their own folder (or with the folder as the working directory).

| Folder | Contents |
|---|---|
| [nfl/](nfl/) | NFL point-spread projections. The `nfl_prod_*` scripts are the production versions: `nfl_prod_projections.py` makes the weekly projections and `nfl_prod_backtest.py` backtests it. The `nflpicks*` scripts are older versions kept for reference. |
| [mlb/](mlb/) | MLB picks, backtest and lines UI. |
| [nba/](nba/) | NBA picks. |
| [college/](college/) | College football and basketball picks. |
| [fantasy/](fantasy/) | Fantasy football drafting and analytics scripts. |
| [fantasy_football_web/](fantasy_football_web/) | Flask site for fantasy league history and analytics (see its own README). |
| [crypto_bot/](crypto_bot/) | Crypto signal bot, paper trading, dashboard and its launcher `.bat` files. State (`paper_portfolio.json`, `bot_overrides.json`, `crypto_bot.log`) lives next to the code. |
| [scratch/](scratch/) | One-off experiments and utilities. |

## NFL projections

```
cd nfl
python nfl_prod_projections.py
```

- Year and week are detected automatically; set `PREDICT_YEAR` / `PREDICT_WEEK` in the `__main__` block to override.
- pro-football-reference blocks scripted requests, so pages saved from a browser into `C:/Users/Bobby/Downloads/NFL_Stats/` are used as a fallback.
- Results are logged and appended to `<year>_nfl_projections.xlsx` on the Desktop. Each run adds its week in week order, so a week run late is inserted after earlier weeks and before later ones. Fair lines already in the sheet are kept, and every run refreshes the actual score difference for all games played so far. Notes are never touched.
- Blank spreads are filled in with DraftKings lines from Action Network's public scoreboard feed (home spread, negative = home favored). If DraftKings has no line for a game, the consensus line (then other books) is used and the run logs which book it took. Spreads already in the sheet are never replaced. Games that have been played get the closing line. Action Network has DraftKings lines from 2024, consensus lines only for 2023, and nothing earlier, so those stay blank.
- Covering Team and Result cells are shaded light green for a win and light red for a loss, and each week's games are boxed with a bold outline.
- The Backtest ATS Win % column shows how often the model's pick covered against the market at that size of disagreement (2023-2025 backtest: about 54% under 3 points, 51% for 3-6, 43% for 6 or more). The model has not beaten the DraftKings/consensus lines in backtesting, and its biggest disagreements did worst, so treat the projections as information to weigh rather than picks to follow. The table behind the column is in O:P of the sheet and the numbers are `ATS_HIT_RATE_BY_EDGE` in the script.

## Research: can any model beat the closing spread?

`nfl_research_ats.py` downloads free [nflverse](https://github.com/nflverse) data (`nflverse_data.py` caches it in `nfl/data_cache/`) and runs a walk-forward test: each season is predicted by a model trained only on earlier seasons, using closing spreads for 2012-2026 (about 3,600 games). It tests EPA team ratings, a regression like the production model on season-to-date yards and turnovers, and gradient boosting on every pre-game feature, and reports ATS win rates by season, edge size and spread size.

```
cd nfl
python nfl_research_ats.py
```

## Crypto bot

Run `crypto_bot\run_crypto_bot.bat` (console) or `crypto_bot\run_all.bat` (bot and dashboard). The scripts use their own folder as the working directory, so they can be launched from anywhere.
