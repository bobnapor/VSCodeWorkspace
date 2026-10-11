# NFL spread picks: current state and handoff

This document describes what the NFL system does today, how it works, what evidence supports it, what was tried and failed, and where it is weak. It is written so a fresh reader can restate the problem from scratch, look at the current state, and decide what to do next. Section 1 states the problem without reference to any solution; sections 2 onward describe the current solution. Read section 1, form your own plan, then compare it with the rest.

Branch state when this was written: `master` holds the earlier "top 3 underdogs a week" version (`f8c86dc`); `feature/model-improvement` (`4c7ee49`, pushed, not merged) holds everything described here, including the key-number pick rule.

---

## 1. The problem (restated neutrally)

The owner bets NFL **point spreads** (DraftKings, flat stakes, standard -110 juice) in **weeks 5-16 of the regular season**. He wants a system that, each week, **decides which games are worth betting** based on criteria, so that he can place those bets without further thought. It is not required to pick a fixed number of games; the number can vary by week (3 or more a week would be nice, not required).

**Success target:** over a long run, the games the system picks should **win about 60% or better against the spread**, judged by the hit rate of all games picked across a season (individual seasons will vary; he understands the law of large numbers). Break-even at -110 is 52.38%. For reference, he previously hand-picked games and won just under 60% of them in earlier seasons (spreadsheets in `C:\Users\Bobby\Documents\2024 NFL Projections` and `Downloads` hold his past picks; an early analysis suggested he favored spreads of 4 or more).

**His starting hypothesis** was that the most predictive stats are few and familiar: yards gained and allowed, first downs gained and allowed, and takeaways/turnovers. He is open to being wrong and to creative approaches.

**Constraints and context**
- Free data only. Pro-Football-Reference now blocks scraping (Cloudflare 403), so the old scrapers no longer work.
- The market (closing spread) is a very strong baseline: its average miss on the final margin is about 9.94 points, better than any model here.
- The owner runs the system on his Windows machine and gets a per-season Excel sheet on his Desktop; he wants the sheet to show the picks clearly and fit on one screen.

## 2. Current state in one page

- **What runs:** `python nfl_prod_projections.py` (from `nfl/`) downloads the free nflverse data, projects the next unplayed week with a stats model, computes a second "correction" model, fetches DraftKings spreads from Action Network, decides which games are **Picks**, and writes/updates `C:\Users\Bobby\OneDrive\Desktop\<year>_nfl_projections.xlsx`.
- **The pick rule (on the feature branch):** a game is a **Pick** when (a) it is a regular-season week 5-16, (b) the **underdog's spread is exactly 3, 6, 7, 10 or 14** (key numbers), and (c) the correction model thinks the underdog covers by at least **0.25 points** (`SIGNAL_THRESHOLD` in `nfl_signal.py`; also an editable cell in the sheet).
- **Backtest of that rule** (walk-forward, 2006-2026, weeks 5-16, bets at the closing spread): 228 picks (about 11 a season, about 0.9 a week), **65.8%** covered (64.5% in 2006-2017, 68.4% in 2018 on), 17 of 20 seasons above break-even, about +27% ROI at the real closing spread odds. With any positive correction instead of 0.25: 361 picks (about 17 a season), 60.4%, +17% ROI. Volume of 3 or more picks a week at 60% is **not** reached by anything tried (see section 6).
- **How confident to be:** the rule was found by scanning about 25 game characteristics after the correction model already existed, so the backtest is optimistic. A fair prior for the true rate is roughly 58-65%. Season-to-season swings are large with about 11 picks a season (worst seasons were 35-47%). Nothing has been forward-tested. See section 7.
- **What does not work:** the owner's gut stats (yards, first downs, turnovers) carry no information beyond the spread; favorites show no edge; the stats-only projection (the original model) picks winners at about 51% against the spread; every richer model family tried landed in the same 54-58% band for broad pick sets.

## 3. How the system works (technical)

### 3.1 Data (all free, cached in `nfl/data_cache/`, which is gitignored)

Loaded by `nflverse_data.py` and `nfl_pbp_features.py`; the latest season is refreshed every run, older seasons use the cache, and a download failure falls back to the cached copy.

| Source | Content |
|---|---|
| `nflverse/nfldata games.csv` | Every game since 1999: scores, **closing `spread_line` (positive = home favored)**, `total_line`, moneylines and spread odds (2008+), weather, roof, surface, coaches, rest days, starting QBs |
| nflverse `stats_team_week_{y}` | Per team-game box score: attempts, carries, sacks, yards, first downs, interceptions, fumbles lost, passing and rushing EPA |
| nflverse `stats_player_week_{y}` | Player-game stats (used for quarterback quality) |
| nflverse `injuries_{y}` (2009+) | Official weekly injury reports |
| nflverse `play_by_play_{y}.csv.gz` | About 19 MB per season; **aggregated per team-game and the raw file deleted**; kept as `pbp_team_{y}.csv` (31 offense stats, with the defense side taken from the opponent) and `pbp_scoring_{y}.csv` (TD/FG/XP/two-point/safety counts per team-game) |
| Action Network scoreboard API | Current DraftKings (book 68; also 1534 PA, 15 consensus, 30 opener) spreads and public betting splits; DraftKings available from 2024, consensus for 2023, nothing earlier |

nflverse uses current team codes (LV, LAC, LA) for every season. Pushes are excluded from all hit-rate statistics (the stake is returned), so "win %" means wins over wins plus losses.

### 3.2 Team ratings (`nfl_lab.rate_teams`, `nfl_model.add_pregame_ratings`)

For each (own stat, allowed stat) pair, each team gets a **pre-game** rating built in date order: a recency-weighted average of its earlier games (decay 0.90 per game, about a 7-game half-life), corrected by 0.6 of each opponent's deviation from the league average, shrunk toward the league mean by 3 pseudo-games early on, and carried into a new season at 60%. Features for a game are home minus away ratings, so everything is known before kickoff and the same code builds training rows and upcoming games. `nfl_lab.py` rates 37 pairs (6 box-score pairs including EPA per play, and 31 play-by-play pairs such as EPA/success by pass/rush/early/late, explosive rate, sack/interception/fumble rates, third-down rate, pace via drives and plays, special-teams EPA); `nfl_model.py` rates the smaller production set.

### 3.3 Model A: the stats-only margin projection (`nfl_model.py`, the original deliverable)

Ridge regression (alpha 300) of the home margin on ratings for **yards, turnovers, first downs, points and plays** (offense and defense), a margin Elo (K 0.07, 30% offseason regression, 2-point home field), rest-day difference, neutral site, and quarterback features (rating difference, injury edge). Trains on the last 12 seasons of completed games before the target week; projections average 200 bootstrap fits. Backtest margin error (2014-2026): 10.19 points (the previous scraped-data model 10.40; the closing spread 9.94). **It does not beat the spread**: it picks the right side about 51% of the time. It is now only a logged reference and the fallback fair line.

### 3.4 Model B: the correction model (`nfl_signal.py`, the thing the picks come from)

Ridge regression, **alpha 10000**, on the cover margin (`resid = result - spread_line`, so positive means the home side covered), with features `PBP_FEATURES + BOX_FEATURES + BASIC_CONTEXT` (the 31 play-by-play and 5 box-score pairs, offense and defense, plus `neutral`, `rest_diff`, `elo_diff`), standardized, trained on the **14 seasons** before the target season (`lab.training_games`, window 14). Its output is the **correction**: points by which it thinks the spread is off, positive meaning the home team covers. It is strongly regularized, so most corrections are small (a correction of 0.5 is already unusual). Production code: `week_corrections(season, week)`. The same function with an expanding walk-forward is the research harness (`lab.walk_forward`).

### 3.5 The pick rule (`nfl_signal.is_pick`, mirrored in the sheet formula)

Underdog = the side getting points at the current DraftKings spread. `toward_underdog` = the correction signed toward that side. Pick iff `5 <= week <= 16`, `abs(spread) in KEY_LINES = (3, 6, 7, 10, 14)` (`nfl_sheet.KEY_LINES`), `toward_underdog > 0`, and `toward_underdog >= SIGNAL_THRESHOLD (0.25)`. Favorites are never picked.

### 3.6 Weekly runner (`nfl_prod_projections.py`)

1. Load/refresh nflverse data, pick the next unplayed week (or `PREDICT_YEAR` / `PREDICT_WEEK` overrides in the `__main__` block).
2. Model A projections (logged to `C:/Users/Bobby/nfl_predictions_<timestamp>.txt` and the console).
3. Build the full research table (`lab.build_table(lab.ALL_PAIRS, refresh=True)`), compute the backtested hit-rate table (`nfl_signal.backtest_hit_rates`, about 30 s), compute this week's corrections and the corrections for earlier weeks already in the sheet (backfill; never adds games).
4. Fetch DraftKings spreads (fallback order: DraftKings NJ, DK PA, consensus, other books, opener; the book used is logged), log the week's picks.
5. Append a timestamped snapshot of lines and betting splits to `data_cache/line_log.csv` (to eventually measure line movement; the public history has only stale openers and closes).
6. Write/refresh the Excel sheet.

### 3.7 The Excel sheet (`nfl_sheet.py`)

One workbook per season, rewritten in full every run and read back **by header text** (so older layouts convert in place; a file missing the basic columns is backed up to `*_old_layout_<timestamp>.xlsx`). Columns, in order: Week, Home Team, Away Team, **Spread (as Home)** (manual/yellow; negative = home favored), **Pick** (formula, shown like "Bills +3", blue highlight), Correction Toward Underdog (formula), Backtest ATS Win % (formula lookup), Fair Line (as Home), Fair Line vs Spread, Side to Cover, **Actual Score Difference (as Home)** (manual/yellow), Covering Team, Result (green/red), QB Injury Flag, Notes (manual/yellow), Year, Model Correction (as Home). A table and three editable settings sit to the right of the data (minimum correction, first and last pick week). Behaviors worth knowing:
- **Fair Line = -spread + correction**, rounded to half points, never rounding the correction to zero, so Side to Cover and Fair Line vs Spread always agree with Model B. Games with no spread yet keep the Model A fair line until one exists.
- A spread already in the sheet is **never overwritten**; for unplayed games the run logs a warning when DraftKings' current line differs (a pick holds only at the key number you can bet).
- The Backtest ATS Win % table is **computed every run** from the walk-forward backtest (weeks 5-16, 2006 to date), split into key-line and other-line columns with the game counts; nothing is typed in.
- Formula columns show team nicknames; columns are autosized, headers wrap, QB flags and notes wrap in a capped width, header row frozen.
- Formulas have been checked by emulating them in Python and by reading back the written workbook; **they have not been opened in Excel in this environment**.

## 4. Evidence and protocol

All model evidence is **walk-forward**: each test season is predicted by a model fit only on earlier seasons. Research is split into a **discovery** period (2006-2017, where choices were made) and a **holdout** (2018 on, a confirmation). Note: that holdout was looked at repeatedly while exploring (see section 7), so it is not pristine. Test games with a push are dropped. ROI uses the real closing spread odds in `games.csv` where available (2008+), else -110.

Headline results (weeks 5-16, 2006-2026):

| Selection | Picks per season | Win % | Notes |
|---|---|---|---|
| Underdog, key line, correction >= 0.25 (**current rule**) | 10.9 | 65.8% | disc 64.5% (n=152), hold 68.4% (n=76); +27.4% ROI; permutation p = 0.005 versus random key-line underdogs the model also likes |
| Underdog, key line, correction > 0 | 17.2 | 60.4% | disc 60.3%, hold 60.6%; +17.2% ROI |
| Underdog, key line, correction >= 0.5 | 6.8 | 66.9% | disc 60.4%, hold 80.4% |
| Underdog, whole-number line of 3+, correction > 0 | 24.3 | 58.9% | |
| Every whole-number line, correction > 0 | 33 | 56.7% | |
| Top 3 underdogs by correction each week, any line (earlier rule, all weeks 1-18) | about 49 | about 57.6% | holdout 58.5% |
| No model: bet every underdog on a key line | about 48 | 52.4% | other whole-number lines 49.2%, half-point lines 50.4% |
| Model Correction on favorites, any model variant | | 48-54% | no edge |

Per-season record of the current rule: 2006 12-4, 2007 12-5, 2008 10-7, 2009 6-2, 2010 10-3, 2011 11-7, 2012 6-5, 2013 7-2, 2014 7-8, 2015 6-3, 2016 7-1, 2017 4-7, 2018 2-2, 2019 7-2, 2020 11-7, 2021 7-2, 2022 2-1, 2023 7-4, 2024 10-2, 2025 6-4 (2026 in progress).

Reproduce: `python nfl_research_select.py production` (the rule), `python nfl_research_select.py` (the condition scan that found the key-number cut), `python nfl_research_select.py favorites`, `python nfl_research_signal.py` (the earlier underdog-signal evidence), `python nfl_research_weekly.py`, `python nfl_research_window.py`, `python nfl_research_features.py [sets singles forward frontier dog consensus]`, `python nfl_research_sim.py`.

## 5. What was found along the way (and why the rule looks like it does)

1. **Model A has no edge against the spread** (it is a decent margin predictor, MAE 10.19 vs the market's 9.94). Any signal must come from where the market misprices.
2. **The correction model's signal is real but small and only on underdogs** (all weeks: underdog-side picks 54.8%, n=1,841, versus 51.3% for underdogs in general; 17 of 21 seasons above 50%; larger corrections do better; a placebo test gave p < 0.002; favorites 50.7%).
3. **The signal concentrates on key-number lines**: among top-quartile underdog picks the key-number condition stood out (64% disc / 67% hold versus about 52% outside); half-point lines showed none. The mechanism is unproven (sticky lines at 3 and 7 and favorites being overpriced by public money are plausible stories, not tested).
4. **A flat frontier**: tightening a size-based cutoff from about 49 to 8 picks a season barely moved the pooled hit rate (55.7% to 56.6%); selectivity by score alone does not concentrate the edge. The concentration comes from the line condition.

## 6. What was tried and rejected (do not redo without a new idea)

- **The owner's gut stats** (yards, first downs, turnovers, plus points/plays): no information beyond the spread on weeks 5-16 (holdout slope negative, about 50% hit rate); the whole box-score family is priced in. First downs add about 0.01 points of margin accuracy (t of 1.9).
- **Few-stat models**: no single stat of 37 clears chance (best t about 2.4); only pace (drives) and success-rate stats lean consistently positive. A forward-selected 4-stat set (drives, special-teams EPA, rushing success rate, interception rate) correlates better with the cover margin on the holdout (r 0.109 versus 0.065 for the full model) but picks **worse** (underdog top-3 52-55% versus 57.6%): a more compact model is not a better betting model.
- **Training window**: 3, 5 or 8 seasons instead of 14: not better (5 seasons 55.6-56.2%; 3 seasons worse). Ratings already fade old seasons via the 60% carry.
- **Agreement filters** (stats model and correction model on the same side): worse (53.9%); averaging models across windows or families: no gain; consensus of full, 4-stat and underdog-frame models: about 57-58% on 3-5 picks a week, within noise.
- **Underdog-frame model** (ridge on the underdog's cover margin using underdog-minus-favorite features plus spread/total shape): same 56-57% band as the original; a no-team-stats control scores 50-55%.
- **Favorite-side models trained on subsets** (key lines only, whole numbers only, lines of 3+): 48-54% on both key lines and all lines.
- **Other features/models**: market prices (moneyline-implied, juice lean), weather/context, cover history, overreaction, travel, ticket/money splits (2023+ only), QB injury flag and injury counts by position group, gradient boosting, logistic classification, a denoised two-stage target: none added reliable information.
- **Opening-line betting** (apparent 61%): an artifact, because the Action Network opener is stale (mean move 2.9 points) so any noisy copy of the close beats it.
- **Monte Carlo** (`nfl_sim.py`, `nfl_research_sim.py`): a drive-based scoring simulator (12 drives, gamma-distributed expected points anchored to market plus correction, TD/FG/XP/two-point/safety, lattice reweighting so key margins match 12 seasons of real results). It reproduces key-number frequency but supplies shape, not information. Simulated-EV selection did not beat the correction-size rule; moneyline underdogs showed no edge (about -1% to -4% ROI); Kelly staking: flat 1% about x1.35 over 15 seasons (18% max drawdown), half Kelly about x2 (46% drawdown); season-block bootstrap of ROI had P(ROI>0) near 0.9 for the broad calibrated-EV rule, but a skeptical prior left only about 36% chance the true rate beats break-even for that rule.

**Not tried:** Kalman/state-space ratings, Bayesian hierarchical models, Poisson/Dixon-Coles team models, neural nets/GPs, player-level (position-group, QB-specific) ratings, tuning the rating hyperparameters (decay, carry, opponent adjustment) specifically for the correction model, new data such as pressure/personnel/PFF-style grades, totals-market angles, line movement/closing-line-value (needs the `line_log.csv` to accumulate), explicit modeling of why key numbers matter (e.g. pricing the push/half-point value).

## 7. Weak points a fresh reviewer should interrogate

1. **Selection bias.** About 25 conditions were scanned and the key-number cut won; thresholds (0.25) and the line set were also looked at on the same data. The discovery/holdout split was reused across roughly a hundred model variants in this effort, so the 2018+ holdout is no longer clean. The honest test is forward: 2026 onward.
2. **Closing line versus bet time.** The backtest bets the **closing** spread; the owner bets earlier in the week. A pick conditions on the line sitting on a key number; if it moves off by close (or between run and bet) the backtest does not apply. The sheet warns when the sheet's spread differs from the current DraftKings line. `line_log.csv` will eventually allow a real test.
3. **Sample size.** About 228 picks over 20 seasons; a 95% interval on the pooled rate is roughly +/-6 points, and a single season has about 11 picks.
4. **Time instability.** The broad model's hit rate was higher in 2018+ (about 59%) than 2006-2017 (about 54-55%) on top-fifth picks; the difference is not statistically separable but the rule may be partly an era effect (play-by-play data quality, rule changes).
5. **Lines used are the nflverse consensus close**, not DraftKings; DraftKings lines, juice and limits differ (the owner's actual juice may be worse than the closing odds in the data).
6. **Volume.** About 0.9 to 1.4 picks a week against a stated wish of 3 or more at 60%; the research found no way to get both.
7. **Spreadsheet formulas untested in Excel**; the Pick formula is a re-implementation of `nfl_signal.is_pick` and could drift from it.
8. **Mechanism unexplained.** Without a causal story for why key-number underdogs are mispriced, regime change is a real risk.

## 8. Repository map (`nfl/`)

Production (what the weekly run uses): `nfl_prod_projections.py` (runner), `nfl_model.py` (Model A), `nfl_signal.py` (Model B and the pick rule), `nfl_sheet.py` (workbook and sheet formulas, DraftKings spreads, QB flags), `nflverse_data.py`, `nfl_pbp_features.py`, `nfl_injury_features.py`, `nfl_lab.py` (rating/feature table and walk-forward harness used by both research and production), `nfl_market_moves.py` (Action Network lines/splits and the line log).

Research (each is runnable and prints its evidence): `nfl_research_select.py` (selection rules, key numbers, favorites), `nfl_research_features.py` (stat sets, single stats, forward selection, frontier, underdog frame, consensus), `nfl_research_weekly.py` (top-N per week), `nfl_research_window.py` (training windows, agreement), `nfl_research_signal.py` (original underdog-signal evidence), `nfl_research_sim.py` and `nfl_sim.py` (Monte Carlo), `nfl_research_model.py`, `nfl_research_ats.py`, `nfl_research_injuries.py`, `nfl_lab_extra.py`, `nfl_prod_backtest.py` (week-by-week replay of Model A).

Legacy (Pro-Football-Reference scrapers; broken since the Cloudflare block, kept for reference): `nflpicks.py`, `nflpicks_backtest.py`, `nflpicks_v2_0.py`, `nflpicks_v2_0_multithreaded.py`.

Key framework functions in `nfl_lab.py`: `default_table()` (cached full table), `build_table(pairs, refresh)`, `walk_forward(table, fit_predict, first_test=2006, last_test=2026, window=14)`, `ridge_margin(columns, alpha, target)` (drops near-constant columns, returns zeros if none remain), `information_test`, `scorecard`; constants `DISCOVERY_END = 2017`, `PBP_FEATURES`, `BOX_FEATURES`, `BASIC_CONTEXT`. The table includes `resid` (cover margin), market features (`abs_spread`, `on_key_number`, `half_point`, `ml_logit`, `juice_lean`), and context (rest, weather, roof, coaches, primetime).

## 9. Environment and gotchas

- Windows; **32-bit Python 3.8.5**; pandas 2.0.3, numpy 1.24.4, scikit-learn 0.24.2, scipy 1.5.2, openpyxl, requests. Not installed: lightgbm, xgboost, statsmodels, torch, pymc. `HistGradientBoosting` needs `from sklearn.experimental import enable_hist_gradient_boosting`. 32-bit numpy `bincount` needs int32 input.
- Run scripts from inside `nfl/` (they import siblings by name). `data_cache/` is rebuilt on demand (about 19 MB per season of play-by-play is downloaded and discarded once aggregated; a full rebuild takes a while). `lab_table_default.pkl` caches the research table; delete it (or call `default_table(refresh=True)`) after changing rating code.
- The workbook path comes from `nfl_sheet.get_desktop_path()` (OneDrive-aware); tests should monkeypatch it to a temp folder so the real sheet is untouched. The runner's log file path (`C:/Users/Bobby/...`) is hardcoded.
- The 2026 season is in progress at the time of writing (week 5 was the latest run).

## 10. Suggested use of this document for the fresh start

1. Read only section 1 and write your own plan: what evidence would convince you a game is worth a bet at about 60%, what you would model, how you would validate it, and what you expect to find.
2. Read sections 2-7 and compare: where does your plan differ, which of your ideas is in section 6 (already rejected) or "Not tried", and which weak points in section 7 your plan would address.
3. Decide the next step by expected value of information, e.g. a forward-test harness and a closing-line-value measurement versus a genuinely new data source versus a structural explanation of the key-number effect, and be explicit about the multiple-comparisons budget you are spending.
