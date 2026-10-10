# VSCodeWorkspace

Personal Python projects, one folder per area. Scripts import their siblings by name, so run them from inside their own folder (or with the folder as the working directory).

| Folder | Contents |
|---|---|
| [nfl/](nfl/) | NFL point-spread projections. `nfl_prod_projections.py` makes the weekly projections and `nfl_prod_backtest.py` backtests it; `nfl_model.py` is the model and `nfl_sheet.py` the spreadsheet. The `nflpicks*` scripts are older versions kept for reference. |
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
- All data comes from the free [nflverse](https://github.com/nflverse) game-by-game files (downloaded on each run and cached in `nfl/data_cache/`; the latest season is refreshed every run), so nothing is scraped from Pro-Football-Reference any more.
- **The model** (`nfl_model.py`) rates every team before each game on six stats, offense and defense separately: net yards, turnovers, **first downs**, points and plays (the full model also tests EPA per play, which added nothing). Ratings are recency-weighted averages that are adjusted for the strength of the opponents faced, shrunk toward the league average early in the season and carried partly across seasons. A ridge regression combines both teams' ratings with a margin-based Elo rating, rest days, neutral site and quarterback quality and injury status to project the home margin. It trains on the last 12 seasons of games played before the week being projected, and the projections average 200 bootstrap fits.
- In a 2014-2026 walk-forward test this projects margins about 0.2 points better than the previous model (average miss 10.19 vs 10.40 points; the closing spread's is 9.94). First downs help only marginally on top of the other stats (removing them costs about 0.01 points). Against the closing spread it still picks winners at about 51%, so it is information to weigh and not a source of picks.
- Results are logged and appended to `<year>_nfl_projections.xlsx` on the Desktop. Each run adds its week in week order, so a week run late is inserted after earlier weeks and before later ones. Fair lines already in the sheet are kept, and every run refreshes the actual score difference for all games played so far. Notes are never touched.
- Blank spreads are filled in with DraftKings lines from Action Network's public scoreboard feed (home spread, negative = home favored). If DraftKings has no line for a game, the consensus line (then other books) is used and the run logs which book it took. Spreads already in the sheet are never replaced. Games that have been played get the closing line. Action Network has DraftKings lines from 2024, consensus lines only for 2023, and nothing earlier, so those stay blank.
- Covering Team and Result cells are shaded light green for a win and light red for a loss, and each week's games are boxed with a bold outline.
- The Backtest ATS Win % column shows how often the model's pick covered against the market at that size of disagreement (2014-2026 walk-forward: about 51% under 3 points, 53% for 3-6, 49% for 6 or more; break-even at -110 is 52.4%). Treat the projections as information to weigh rather than picks to follow. The table behind the column is in O:P of the sheet and the numbers are `ATS_HIT_RATE_BY_EDGE` in `nfl_sheet.py`.

- The QB Injury Flag column (N) lists games where a team's previous starting quarterback is on this week's injury report (Out, Doubtful or Questionable) and games where the schedule expects a different starter than last week. It comes from the free nflverse injury reports, is refreshed on every run for games not yet played, and is kept as-is once a game has been played. It is information to read, not a signal the projections use: in backtesting it did not predict covers (see below).

## Research: can any model beat the closing spread?

`nfl_research_model.py` is the test behind the production model: each season is predicted by a model trained only on earlier seasons, and it compares the previous production model, each stat set (yards and turnovers, plus first downs, points, plays, EPA), Elo, quarterback features and the closing spread, with paired significance tests. `nfl_prod_backtest.py` replays seasons week by week with the production model itself.

`nfl_research_ats.py` runs the same kind of walk-forward test against closing spreads for 2012-2026 (about 3,600 games) with other model types, including gradient boosting, and reports ATS win rates by season, edge size and spread size (including spreads of 4 and up).

`nfl_research_injuries.py` layers pre-game injury features from `nfl_injury_features.py` (quarterback status from the official reports, quarterback quality, and counts of players Out or Doubtful by position group) onto that test and looks directly at games where a starting quarterback is out.

```
cd nfl
python nfl_research_model.py
python nfl_research_ats.py
python nfl_research_injuries.py
```

## Crypto bot

Run `crypto_bot\run_crypto_bot.bat` (console) or `crypto_bot\run_all.bat` (bot and dashboard). The scripts use their own folder as the working directory, so they can be launched from anywhere.
