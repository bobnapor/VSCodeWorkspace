# NFL spread picks: current state and handoff

This document describes what the NFL system does today, how it works, what evidence supports it, what was tried and failed, and where it is weak. It is written so a fresh reader can restate the problem from scratch, look at the current state, and decide what to do next. Section 1 states the problem without reference to any solution; sections 2 onward describe the current solution. Read section 1, form your own plan, then compare it with the rest.

Branch state (October 11, 2026): `master` holds an earlier "top 3 underdogs a week" version (`f8c86dc`). `feature/model-improvement` (pushed, not merged) holds everything described here: the validated rules, the untouched-data test that overturned the earlier key-number rule, and teaser legs.

---

## 1. The problem (restated neutrally)

The owner bets NFL **point spreads** (DraftKings, flat stakes, standard -110 juice) in **weeks 5-16 of the regular season**. He wants a system that, each week, **decides which games are worth betting** based on criteria, so that he can place those bets without further thought. It is not required to pick a fixed number of games; the number can vary by week (3 or more a week would be nice, not required).

**Success target:** over a long run, the games the system picks should **win about 60% or better against the spread**, judged by the hit rate of all games picked across a season (individual seasons will vary). Break-even at -110 is 52.38%. For reference, he previously hand-picked games and won just under 60% of them in earlier seasons (spreadsheets in `C:\Users\Bobby\Documents\2024 NFL Projections` and `Downloads`; an early analysis suggested he favored spreads of 4 or more).

**His starting hypothesis** was that the most predictive stats are few and familiar: yards gained and allowed, first downs gained and allowed, and takeaways/turnovers. He is open to being wrong and to creative approaches.

**Constraints and context**
- Free data only. Pro-Football-Reference now blocks scraping (Cloudflare 403), so the old scrapers no longer work.
- The market (closing spread) is a very strong baseline: its average miss on the final margin is about 9.94 points, better than any model here.
- The owner runs the system on his Windows machine and gets a per-season Excel sheet on his Desktop; he wants the sheet to show the picks clearly and fit on one screen.

## 2. Current state in one page

- **What runs:** `python nfl_prod_projections.py` (from `nfl/`) downloads the free nflverse data, projects the next unplayed week with a stats model, computes a second "correction" model, fetches DraftKings spreads from Action Network, flags **Picks** and **Teaser Legs**, and writes/updates `C:\Users\Bobby\OneDrive\Desktop\<year>_nfl_projections.xlsx`.
- **Straight picks:** in weeks 5-16, **every underdog the correction model favors** (any positive correction toward the underdog; `nfl_signal.is_pick`). About 4-5 a week.
- **Evidence:** those underdogs covered **53.7% on 1999-2005**, **55.4% on 2006-2017** and **55.0% on 2018-2026** (weeks 5-16, pushes excluded, about +7% ROI at the real closing odds). 1999-2005 is the clean test: no research step had ever scored those seasons before the validation in October 2026.
- **Honest expectation: about 55%, not 60%.** The correction's information about the cover margin is real and stable (correlation 0.06-0.08 in every era, t of about 2.5 each), but a signal that size supports about 55-56% at any volume. A 60% rule would need a correlation of about 0.11 for one pick a week, 0.14 for three. With about 60 picks a season at 55%, roughly one season in three still loses money.
- **Teaser legs:** the published Wong rule (2001): underdogs of +1.5/+2.5 teased 6 points to +7.5/+8.5, favorites of -7.5/-8.5 teased to -1.5/-2.5. Legs won **75.7% (underdogs) and 74.9% (favorites) since 1999**, weeks 5-16; since 2018 underdog legs 76.1% but favorite legs only 70.8%. A 2-team teaser needs 72.4% per leg at -110 and 73.9% at -120 and wins about 55-58% as a single bet. The value depends on the book's teaser price.
- **What was overturned:** the previous rule (underdog on a 3/6/7/10/14 line with correction >= 0.25) showed 65.8% on 2006-2026 but went **53-50 (51.5%) on the untouched seasons**. It was a selection artifact (found by scanning about 25 conditions on the same data), and the half-point-line pattern even reversed there.

## 3. How the system works (technical)

### 3.1 Data (all free, cached in `nfl/data_cache/`, which is gitignored)

Loaded by `nflverse_data.py` and `nfl_pbp_features.py`; the latest season is refreshed every run, older seasons use the cache, and a download failure falls back to the cached copy.

| Source | Content |
|---|---|
| `nflverse/nfldata games.csv` | Every game since 1999: scores, **closing `spread_line` (positive = home favored)**, `total_line`, moneylines and spread odds (2006+, complete from 2010), weather, roof, surface, coaches, rest days, starting QB ids and names (all seasons) |
| nflverse `stats_team_week_{y}` | Per team-game box score: attempts, carries, sacks, yards, first downs, interceptions, fumbles lost, passing and rushing EPA |
| nflverse `stats_player_week_{y}` | Player-game stats (used for quarterback quality) |
| nflverse `injuries_{y}` (2009+) | Official weekly injury reports |
| nflverse `play_by_play_{y}.csv.gz` | About 19 MB per season; **aggregated per team-game and the raw file deleted**; kept as `pbp_team_{y}.csv` (31 offense stats, with the defense side taken from the opponent) and `pbp_scoring_{y}.csv` (TD/FG/XP/two-point/safety counts). Air yards, CPOE and pass rate over expected start in 2006; the others go back to 1999 |
| Action Network scoreboard API | Current DraftKings (book 68; also 1534 PA, 15 consensus, 30 opener) spreads and public betting splits; DraftKings available from 2024, consensus for 2023, nothing earlier |

nflverse uses current team codes (LV, LAC, LA) for every season. Pushes are excluded from all hit-rate statistics (the stake is returned), so "win %" means wins over wins plus losses.

### 3.2 Team ratings (`nfl_lab.rate_teams`, `nfl_model.add_pregame_ratings`)

For each (own stat, allowed stat) pair, each team gets a **pre-game** rating built in date order: a recency-weighted average of its earlier games (decay 0.90 per game, about a 7-game half-life), corrected by 0.6 of each opponent's deviation from the league average, shrunk toward the league mean by 3 pseudo-games early on, and carried into a new season at 60%. Features for a game are home minus away ratings, so everything is known before kickoff and the same code builds training rows and upcoming games. `nfl_lab.py` rates 37 pairs (6 box-score pairs including EPA per play, and 31 play-by-play pairs such as EPA/success by pass/rush/early/late, explosive rate, sack/interception/fumble rates, third-down rate, pace via drives and plays, special-teams EPA). Elo (`nm.add_elo`) is also pre-game. Audited in October 2026: no look-ahead found (the only global quantity is the league mean used for shrinkage, which cancels in home-minus-away differences).

### 3.3 Model A: the stats-only margin projection (`nfl_model.py`, the original deliverable)

Ridge regression (alpha 300) of the home margin on ratings for **yards, turnovers, first downs, points and plays** (offense and defense), a margin Elo, rest-day difference, neutral site and quarterback features. Trains on the last 12 seasons; projections average `BOOTSTRAP_RUNS` bootstrap fits. Margin error (2014-2026): 10.19 points (the closing spread 9.94). **It does not beat the spread** (about 51% against it). It is only a logged reference and the fallback fair line.

### 3.4 Model B: the correction model (`nfl_signal.py`, the thing the picks come from)

Ridge regression, **alpha 10000**, on the cover margin (`resid = result - spread_line`; positive = home covered), with features `PBP_FEATURES + BOX_FEATURES + BASIC_CONTEXT` (the 31 play-by-play and 5 box-score pairs, offense and defense, plus `neutral`, `rest_diff`, `elo_diff`), standardized, trained on the **14 seasons** before the target season. Its output is the **correction**: points by which it thinks the spread is off, positive meaning the home team covers. Strongly regularized, so most corrections are small (|correction| above 0.5 is uncommon). Production code: `week_corrections(season, week)`. `scored_history(table, season)` scores every season out of sample: 2006 on walk-forward, 1999-2005 by a model fitted on 2006-2019.

### 3.5 The rules (`nfl_signal.is_pick`, `nfl_signal.teaser_leg`, mirrored in the sheet formulas)

- **Pick:** `5 <= week <= 16` and `toward_underdog > 0` (and `>= SIGNAL_THRESHOLD`, default 0; the sheet setting "Minimum correction toward underdog" overrides it in the sheet). Underdog = the side getting points at the spread in the sheet; `toward_underdog` = the correction signed toward that side. Favorites are never picked straight.
- **Teaser leg:** `5 <= week <= 16` and the spread is 1.5/2.5 (the underdog, teased to +7.5/+8.5) or 7.5/8.5 (the favorite, teased to -1.5/-2.5). No model involved.

### 3.6 Weekly runner (`nfl_prod_projections.py`)

1. Load/refresh nflverse data, pick the next unplayed week (or `PREDICT_YEAR` / `PREDICT_WEEK` overrides in the `__main__` block).
2. Model A projections (logged to `C:/Users/Bobby/nfl_predictions_<timestamp>.txt` and the console).
3. Build the full research table, compute the hit-rate table (`backtest_hit_rates`, about 30 s) and the teaser-leg rates, this week's corrections, and corrections for earlier weeks already in the sheet (backfill; never adds games).
4. Fetch DraftKings spreads (fallback: DraftKings NJ, DK PA, consensus, other books, opener; the book used is logged), log the week's picks and teaser legs.
5. Append a timestamped snapshot of lines and betting splits to `data_cache/line_log.csv`.
6. Write/refresh the Excel sheet.

### 3.7 The Excel sheet (`nfl_sheet.py`)

One workbook per season, rewritten in full every run and read back **by header text** (older layouts convert in place; a file missing the basic columns is backed up to `*_old_layout_<timestamp>.xlsx`). Columns: Week, Home Team, Away Team, **Spread (as Home)** (manual/yellow; negative = home favored), **Pick** (light blue, e.g. "Bills +3"), **Teaser Leg (6 pts)** (light orange, e.g. "Saints +8.5"), Correction Toward Underdog, Backtest ATS Win %, Fair Line (as Home), Fair Line vs Spread, Side to Cover, **Actual Score Difference (as Home)** (manual/yellow), Covering Team, Result, Teaser Result, QB Injury Flag, Notes (manual/yellow), Year, Model Correction (as Home). Right of the data: the hit-rate table (correction toward underdog from, rate, games; 1999 to date, weeks 5-16), three editable settings (minimum correction toward underdog, first and last pick week), and the teaser-leg table (since 1999, since 2018). Behaviors worth knowing:
- **Fair Line = -spread + correction**, in half points, never rounding the correction to zero, so Side to Cover always agrees with Model B. Side to Cover on a non-Pick is not a bet.
- A spread already in the sheet is **never overwritten**; for unplayed games the run logs a warning when DraftKings' current line differs (a pick or teaser leg holds only at the line you can bet).
- Formula columns show team nicknames; columns are autosized, headers wrap, QB flags and notes wrap in a capped width, header row frozen.
- Formulas were checked by emulating them in Python against the code's rules (including 648 synthetic teaser outcomes); **they have not been opened in Excel in this environment**.

## 4. Evidence and protocol

All model evidence is out of sample: 2006 on is walk-forward (each test season predicted by a model fit only on earlier seasons); **1999-2005 is scored by a model fit on 2006-2019** and had never been used by any research step before October 2026, so it is the one clean test. Test games with a push are dropped. ROI uses the real closing spread odds where available (2006+), else -110.

Main results (weeks 5-16):

| Selection | Per season | 1999-2005 (untouched) | 2006-2017 | 2018-2026 |
|---|---|---|---|---|
| **Every underdog the correction favors (current Pick rule)** | 54-68 | **53.7%** (n=473) | 55.4% (n=749) | 55.0% (n=482) |
| No model: every underdog | 150-170 | 50.8% | 50.2% | 51.3% |
| Correction's side, \|correction\| >= 0.5, both sides | 74-86 | 53.1% | 52.9% | 52.9% |
| Old key-number rule (key-line underdog, correction >= 0.25) | 8-15 | **51.5%** (n=103) | 64.5% | 68.4% |
| No model: every underdog on a key line | 40-59 | 49.4% | 52.1% | 52.9% |
| Top 3 underdogs a week by correction (old master rule) | 31-36 | 52.2% | 56.5% | 57.6% |
| Wong teaser legs (no model), all | 20-37 | 75.7% | 76.3% | 74.4% |

Information of the correction (all games, weeks 5-16): correlation with the cover margin 0.084 / 0.056 / 0.064, regression slope t = +2.7 / +2.5 / +2.3. Hit rate does **not** rise reliably with the size of the correction (underdog side by size bins: 53-59% in most bins; the largest corrections, >= 1.0, went 14-29 on 1999-2005).

Selection-process test: re-running the "scan 78 rules, keep the best on earlier seasons" process walk-forward on 2006-2026 picks the key-number rule every season from 2011 and keeps hitting 64% there, so the rule was stable within 2006-2026 but did not carry to 1999-2005. Empirical-Bayes shrinkage across the scanned conditions put it at about 60% even before the untouched test.

Reproduce: `python nfl_research_validate.py` (untouched test, process test, shrinkage, information by era, side and size, robust rule, the 60% requirement), `... models` (production versus the 4-stat model), `... threshold` (both-sides rules with thresholds set from earlier seasons), `... teasers`.

## 5. What was found along the way

1. **Model A has no edge against the spread** (MAE 10.19 vs the market's 9.94).
2. **The correction model carries a small, stable signal** (section 4). Its underdog picks beat the no-model underdog baseline by about 3-5 points in every era; its favorite picks do not (about 51%). The underdog/favorite asymmetry held on the untouched seasons too (53.7% vs 51.2%), consistent with the market shading lines toward favorites.
3. **The key-number concentration was noise.** It looked strong in 2006-2026 (and survived a nested selection test there) but disappeared on 1999-2005, where half-point lines did best instead.
4. **The size of the achievable edge is set by the signal's information.** In a bivariate-normal picture, a signal correlated rho with the cover margin, betting its side on the games where it is strongest, gives: rho 0.07 -> about 56% on 12 picks a season, 55% on 24-36. 60% needs rho about 0.11 (12 picks), 0.13 (24), 0.14 (36). Every honest rule tested landed where this predicts.
5. **A 4-stat model is more informative but not more profitable.** Forward-selected on 2006-2017 only: offense drives per game, special-teams EPA, rushing success rate, interception rate. Pre-registered test on the two eras its selection never saw: correlation 0.102 vs 0.084 (1999-2005) and 0.110 vs 0.064 (2018-2026); binary correlation 0.065 vs 0.036 and 0.063 vs 0.032. But with thresholds set from earlier seasons its both-sides picks hit 54.5% pooled (60.3% / 53.2% / 54.0% at about 2 a week), the same band as the production rule, so production was not switched. Its coefficients read as market blind spots: more drives per game -> covers less (the market overrates per-game volume), better special teams -> covers more, better rushing success and more interceptions forced -> covers more.

## 6. What was tried and rejected (do not redo without a new idea)

- **The owner's gut stats** (yards, first downs, turnovers, plus points/plays): no information beyond the spread in weeks 5-16.
- **Key-number rule** (underdogs on 3/6/7/10/14 lines): failed the untouched test (51.5%).
- **Quarterback-change filter** (skip games where either team's starter differs from its previous game; the team ratings go stale): helped on 1999-2005 (45.7% for underdogs whose starter changed), hurt on 2006-2017 and 2018-2026 (those games hit 56-75%). Not adopted.
- **Size thresholds and top-N-per-week rules** on either model: 52-56% wherever tested out of sample; no threshold concentrates the edge reliably.
- **Training window** 3, 5 or 8 seasons instead of 14: not better.
- **Agreement filters, consensus and averaging of models**: no gain (the production + 4-stat average landed between them).
- **Underdog-frame models, favorite-side models trained on subsets**: 48-58%, within noise; favorites 48-54%.
- **Other features/models**: market prices (moneyline-implied, juice lean), weather/context, cover history, overreaction, travel, ticket/money splits (2023+ only), QB injury flag and injury counts, gradient boosting, logistic classification, a denoised two-stage target: none added reliable information.
- **Opening-line betting** (apparent 61%): an artifact of stale Action Network openers.
- **Monte Carlo** (`nfl_sim.py`, `nfl_research_sim.py`): drive-based scoring simulator reproducing key numbers; simulated-EV selection did not beat simple rules; moneyline underdogs no edge; Kelly: flat 1% about x1.35 over 15 seasons (18% max drawdown), half Kelly about x2 (46% drawdown).

**Not tried:** Kalman/state-space ratings, Bayesian hierarchical models, Poisson/Dixon-Coles team models, neural nets/GPs, player-level ratings (QB-specific EPA that follows the actual starter would fix stale ratings properly rather than filtering), tuning the rating hyperparameters for the correction model, new data (pressure, personnel, charting), totals-market angles, combining the model with teaser legs, line movement/closing-line value (needs `line_log.csv` to accumulate).

## 7. Weak points a fresh reviewer should interrogate

1. **Forking paths are mostly spent.** 2006-2026 has been examined in hundreds of ways; 1999-2005 has now been used once for validation (and a handful of diagnostics). The next clean evidence is forward: 2026 weeks 5-16 onward. Keep any new idea pre-registered and judged on information (correlation), which has far more statistical power than hit rates on small pick sets.
2. **Closing line versus bet time.** Backtests bet the closing spread; the owner bets earlier in the week. Whether the picks beat the close (closing-line value) is the fastest real-world test of an edge and is not yet measured; `line_log.csv` and the nflverse close make it possible (a "closing spread" column in the sheet would do it).
3. **Lines are the nflverse consensus close**, not DraftKings; DraftKings lines, juice and limits differ. Teaser value depends entirely on the book's teaser price (-120 or better needed; -130 removes the edge).
4. **Favorite teaser legs have weakened** (70.8% since 2018, below break-even); underdog legs have not. That split was seen after the fact, so it is noted rather than acted on.
5. **The 60% target.** No rule tested reaches it out of sample; the measured information supports about 55%. Volume at 55% is plentiful (4-5 a week), which matters more for profit than a higher rate on fewer picks.
6. **Spreadsheet formulas untested in Excel**; the Pick and Teaser formulas re-implement `nfl_signal.is_pick` / `teaser_leg` and could drift from them.

## 8. Repository map (`nfl/`)

Production: `nfl_prod_projections.py` (runner), `nfl_model.py` (Model A), `nfl_signal.py` (Model B, the pick and teaser rules, the backtest tables), `nfl_sheet.py` (workbook and formulas, DraftKings spreads, QB flags), `nflverse_data.py`, `nfl_pbp_features.py`, `nfl_injury_features.py`, `nfl_lab.py` (rating/feature table and walk-forward harness), `nfl_market_moves.py` (Action Network lines/splits and the line log).

Research (each runnable, prints its evidence): **`nfl_research_validate.py`** (the untouched-data validation, model comparison, threshold rules, teasers; start here), `nfl_research_select.py` (the condition scan that produced the now-rejected key-number rule), `nfl_research_features.py` (stat sets, single stats, forward selection, frontier, underdog frame, consensus), `nfl_research_weekly.py`, `nfl_research_window.py`, `nfl_research_signal.py` (original underdog-signal evidence), `nfl_research_sim.py` and `nfl_sim.py` (Monte Carlo), `nfl_research_model.py`, `nfl_research_ats.py`, `nfl_research_injuries.py`, `nfl_lab_extra.py`, `nfl_prod_backtest.py`.

Legacy (Pro-Football-Reference scrapers, broken since the Cloudflare block): `nflpicks.py`, `nflpicks_backtest.py`, `nflpicks_v2_0.py`, `nflpicks_v2_0_multithreaded.py`.

Key framework functions in `nfl_lab.py`: `default_table()` (cached full table, 1999 on), `build_table(pairs, refresh)`, `walk_forward(table, fit_predict, first_test=2006, last_test=2026, window=14)`, `ridge_margin(columns, alpha, target)` (drops near-constant columns, returns zeros if none remain), `information_test`, `scorecard`; constants `DISCOVERY_END = 2017`, `PBP_FEATURES`, `BOX_FEATURES`, `BASIC_CONTEXT`.

## 9. Environment and gotchas

- Windows; **32-bit Python 3.8.5**; pandas 2.0.3, numpy 1.24.4, scikit-learn 0.24.2, scipy 1.5.2, openpyxl, requests. Not installed: lightgbm, xgboost, statsmodels, torch, pymc.
- Run scripts from inside `nfl/`. `lab_table_default.pkl` caches the research table; delete it (or `default_table(refresh=True)`) after changing rating code.
- The workbook path comes from `nfl_sheet.get_desktop_path()` (OneDrive-aware); tests should monkeypatch it to a temp folder and copy the real sheet there. The runner's log path (`C:/Users/Bobby/...`) is hardcoded. `BOOTSTRAP_RUNS` in `nfl_prod_projections.py` is the owner's setting (he raised it to 10000 locally).

## 10. Suggested next steps

1. **Measure live performance properly**: add a closing-spread column and closing-line value for every pick and teaser leg, and track the picks from 2026 week 5 on. Closing-line value tells within a season or two what win rate needs many seasons to show.
2. **Look for genuinely new information**, judged by the correlation of a pre-registered signal with the cover margin in every era, with about 0.11+ as the bar for a 60% rule. Candidates with a football rationale: quarterback-specific ratings that follow the actual starter; the 4-stat model's blind spots (pace, special teams) built out properly; charting/pressure data if a free source exists.
3. **Teasers**: if the book prices 2-team 6-point teasers at -120 or better, pairing underdog legs is the best-supported bet in the project; whether the model can filter legs is untested and would need a pre-registered test.
