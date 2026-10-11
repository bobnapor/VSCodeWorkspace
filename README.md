# VSCodeWorkspace

Personal Python projects, one folder per area. Scripts import their siblings by name, so run them from inside their own folder (or with the folder as the working directory).

| Folder | Contents |
|---|---|
| [nfl/](nfl/) | NFL point-spread picks. **See [nfl/README.md](nfl/README.md) for the full current state, evidence and handoff notes.** `nfl_prod_projections.py` runs the weekly projections and picks; `nfl_model.py` and `nfl_signal.py` are the models and `nfl_sheet.py` the spreadsheet. The `nflpicks*` scripts are older (broken) versions kept for reference. |
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
- Results are logged and appended to `<year>_nfl_projections.xlsx` on the Desktop. Each run adds its week in week order, so a week run late is inserted after earlier weeks and before later ones. Every run refreshes the actual score difference for all games played so far. Notes are never touched.
- Blank spreads are filled in with DraftKings lines from Action Network's public scoreboard feed (home spread, negative = home favored). If DraftKings has no line for a game, the consensus line (then other books) is used and the run logs which book it took. Spreads already in the sheet are never replaced. Games that have been played get the closing line. Action Network has DraftKings lines from 2024, consensus lines only for 2023, and nothing earlier, so those stay blank.
- The sheet is laid out so it fits on one screen: the columns you act on come first (Week, Home, Away, Spread, **Pick**, **Teaser Leg**, then the model numbers), then Actual, Covering Team, Result and Teaser Result, and the long QB Injury Flag and Notes plus Year and the raw Model Correction last. Columns are sized to what they show (header and long text wrap), the header row is frozen, and the derived columns show team nicknames, with picks written like "Bills +3" and teaser legs like "Saints +8.5". The sheet is rewritten every run and read back by column header, so a sheet in an older layout is converted in place; the yellow columns (Spread, Actual, Notes) are the ones to edit by hand.
- Covering Team, Result and Teaser Result cells are shaded light green for a win and light red for a loss, picks light blue, teaser legs light orange, and each week's games are boxed with a bold outline.
- The Fair Line in the sheet is the spread moved by the model's correction (see the picks below), so Side to Cover and Fair Line vs Spread always agree with the model. The stats-only projection above is still logged but had no edge against the spread (it picks winners at about 51%), so it no longer sets the sheet's fair line. If no spread is available when you run, the stats-only line is used until one is. Every run recomputes the fair line from each game's spread and correction, and earlier weeks already in the sheet that predate the signal get their corrections filled in (they use only pre-game data). Side to Cover on a game that is not a Pick is not a bet.
- The Backtest ATS Win % column shows how often the model's side covered at that size of correction toward the underdog, over every game from 1999 to date in weeks 5-16, each scored by a model that never saw it (break-even at -110 is 52.4%). The table is in the three columns just right of the data, with the number of games behind each rate, the editable settings and the teaser-leg rates below it. It is recomputed every run (`nfl_signal.backtest_hit_rates`, `nfl_signal.teaser_leg_rates`); nothing in it is typed in.
- **Picks.** A game is a *Pick* when it is a regular-season week 5-16 (the "First pick week" and "Last pick week" settings) and the second model (`nfl_signal.py`, a ridge fit on play-by-play and box-score team ratings) thinks the underdog covers, by more than the "Minimum correction toward underdog" setting (default 0). That is about 4-5 picks a week. Those underdogs covered **53.7% on 1999-2005, 55.4% on 2006-2017 and 55.0% on 2018-2026** (1999-2005 was never used by any research step, so it is the clean test); about +7% ROI at the real closing odds. Expect about 55%, not 60%: the model's information is real and the same in every era, but it is only strong enough for about 55% (60% would need a signal roughly twice as informative), and with about 60 picks a season about one season in three will still lose money. A narrower key-number rule that looked like 65.8% on 2006-2026 went 53-50 (51.5%) on the untouched seasons and was dropped; see `nfl/README.md` and `python nfl_research_validate.py`.
- **Teaser legs.** The *Teaser Leg* column flags the published Wong teaser legs: underdogs of +1.5/+2.5 teased 6 points to +7.5/+8.5, and favorites of -7.5/-8.5 teased to -1.5/-2.5 (pair two of them in one 2-team, 6-point teaser). Legs won about 75% in every era since 1999, weeks 5-16 (favorite legs 71% since 2018); a 2-team teaser needs 72.4% per leg at -110 and 73.9% at -120, and wins about 55-58% as one bet. Check your book's teaser price: at -130 or worse the edge is gone.
- The weekly run warns when the sheet's spread for an unplayed game differs from the current DraftKings line; a pick or teaser leg holds only at the line you can actually bet, and the backtests bet the closing line.
- Once a game is played the sheet fills in its **Closing Spread** and the **Pick Line vs Close** (how many more points the pick got than the closing line; positive means you beat the close), and the **"This season so far"** block below the tables keeps the pick record and win %, the teaser-leg record and win %, and the picks' average line value, so you can see the season's hit rate at a glance.
- Each run also appends the current lines and public betting splits to `nfl/data_cache/line_log.csv` with a timestamp, so there will eventually be real data on how far lines move between when you bet and the close (the public history only has stale openers and the close).

- The QB Injury Flag column lists games where a team's previous starting quarterback is on this week's injury report (Out, Doubtful or Questionable) and games where the schedule expects a different starter than last week. It comes from the free nflverse injury reports, is refreshed on every run for games not yet played, and is kept as-is once a game has been played. It is information to read, not a signal the projections use: in backtesting it did not predict covers (see below).

## Research: can any model beat the closing spread?

`nfl_research_signal.py` is the evidence for the underdog signal (information test, underdog versus favorite picks against the no-model baseline, confidence, thresholds, a nested walk-forward that picks its own settings each season, and a placebo test). It runs on `nfl_lab.py`, a research framework with ratings on 37 box-score and play-by-play stats back to 1999 (`nfl_pbp_features.py` builds the play-by-play aggregates), market prices, weather and context features, and walk-forward scoring tools that report a discovery period (through 2017) and a holdout period separately. `nfl_lab_extra.py` (cover history, travel) and `nfl_market_moves.py` (opening lines and betting splits) hold the other angles that were tested and found to add nothing usable.

`nfl_research_model.py` compares the margin model with the previous production model, each stat set (yards and turnovers, plus first downs, points, plays, EPA), Elo and quarterback features, with paired significance tests. `nfl_prod_backtest.py` replays seasons week by week with the production model.

`nfl_research_ats.py` runs the same kind of walk-forward test against closing spreads for 2012-2026 (about 3,600 games) with other model types, including gradient boosting, and reports ATS win rates by season, edge size and spread size (including spreads of 4 and up).

`nfl_research_injuries.py` layers pre-game injury features from `nfl_injury_features.py` (quarterback status from the official reports, quarterback quality, and counts of players Out or Doubtful by position group) onto that test and looks directly at games where a starting quarterback is out.

`nfl_sim.py` is a Monte Carlo game simulator: each team gets a number of scoring drives, touchdowns and field goals are drawn from binomials with era-based extra-point and two-point rates, and expected points are centered on the closing line plus the signal's correction. Simulated margins are reweighted so the key numbers (3, 7, ties, pushes) match the last 12 seasons of real results. `nfl_research_sim.py` uses it to test selecting bets by simulated expected value (it did not beat the simple correction-size rule), moneyline bets (no edge), Kelly staking, and a season-level bootstrap, Bayesian posterior and bankroll simulation of the signal's uncertainty (true win rate roughly 52-55%, so losing seasons are common). The simulator is research-only and does not feed the spreadsheet.

`nfl_research_features.py` asks which stats carry information the spread does not, scored on weeks 5-16 with stats chosen on 2006-2017 and confirmed on 2018 onward: your gut set (yards, first downs, turnovers) and the whole box-score family carry none (the market already prices them), no single stat clears chance on its own (the best of 37 is t of about 2.4), and the whole edge sits in the 31 play-by-play stats taken together. A forward selection finds a 4-stat set (drives, special-teams EPA, rushing success rate, interception rate) that correlates better with cover margin on the holdout (r 0.11 versus 0.065), but it picks worse (underdog top 3 a week 52-55% versus 57.6% for the full model), so a smaller model is not a better betting model. An underdog-frame model and a consensus of several models land in the same 56-58% band; requiring two models to agree is the best of them (about 57-58% on 3-5 picks a week in both periods), which is within noise of the current signal after the number of variations tried.

`nfl_research_validate.py` is the honest check of all of the above (October 2026): it scores the 1999-2005 seasons, which no research step had used, with a model fitted on 2006-2019. The key-number rule failed there (51.5%), the quarterback-change filter helped only there, and the correction's information turned out stable in every era (correlation with the cover margin 0.06-0.08), which supports about 55% and not 60%. The 4-stat model above passed a pre-registered test on information (0.10-0.11 versus 0.06-0.08 in the two eras its selection never saw) but its picks also land at about 54-55%. It also tests the published Wong teaser rule (legs about 75% in every era). See `nfl/README.md` for the full write-up.

`nfl_research_data.py` tests data the model does not use (October 2026), each against a rule fixed before running: 15 seasons of opening and closing lines from Sportsbook Reviews Online's free archive (`nfl_market_history.py` downloads and parses them), betting at the open instead of the close, a rating of each game's actual starting quarterback, record and luck, pass rush and coverage (QB hits and passes defended), team popularity (primetime games), and retuned rating memory. None added information beyond the closing line, and none produced an edge on favorites: public information is already in the line.

```
cd nfl
python nfl_research_data.py open early qb luck pressure popularity ratings
python nfl_research_validate.py
python nfl_research_validate.py models
python nfl_research_validate.py threshold
python nfl_research_validate.py teasers
python nfl_research_signal.py
python nfl_research_sim.py
python nfl_research_weekly.py
python nfl_research_window.py
python nfl_research_features.py sets singles forward frontier dog consensus
python nfl_research_select.py production
python nfl_research_model.py
python nfl_research_ats.py
python nfl_research_injuries.py
```

## Crypto bot

Run `crypto_bot\run_crypto_bot.bat` (console) or `crypto_bot\run_all.bat` (bot and dashboard). The scripts use their own folder as the working directory, so they can be launched from anywhere.
