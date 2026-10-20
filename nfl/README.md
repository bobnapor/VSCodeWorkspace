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

- **What runs:** `python nfl_prod_projections.py` (from `nfl/`) downloads the free nflverse data, projects the next unplayed week with a stats model, computes a second "correction" model and a plain-words reason for every game's lean, fetches DraftKings spreads from Action Network, flags **Picks**, **Favorite Picks** and **Teaser Legs**, and writes/updates `C:\Users\Bobby\OneDrive\Desktop\<year>_nfl_projections.xlsx` (tabs: This Week, Season Log, Model Info). All three weekly flows (spread projections, player props, wind unders) run with one click from `run_all.bat` or the "NFL weekly run" Desktop shortcut (section 13).
- **Straight picks:** in weeks 5-16, **every underdog the correction model favors** (any positive correction toward the underdog; `nfl_signal.is_pick`). About 4-5 a week.
- **Evidence:** those underdogs covered **53.7% on 1999-2005**, **55.4% on 2006-2017** and **55.0% on 2018-2026** (weeks 5-16, pushes excluded, about +7% ROI at the real closing odds). 1999-2005 is the clean test: no research step had ever scored those seasons before the validation in October 2026.
- **Honest expectation: about 55%, not 60%.** The correction's information about the cover margin is real and stable (correlation 0.06-0.08 in every era, t of about 2.5 each), but a signal that size supports about 55-56% at any volume. A 60% rule would need a correlation of about 0.11 for one pick a week, 0.14 for three. With about 60 picks a season at 55%, roughly one season in three still loses money.
- **Favorite picks (added October 20, 2026; the underdog rule is unchanged):** in weeks 5-16, **night games (kickoff 7 pm Eastern or later: Thursday, Sunday and Monday nights) where the correction leans to the favorite by 0.5 points or more** (`nfl_signal.is_favorite_pick`), about 1 a week. Found by a pre-registered selection process over 78 favorite rules (`nfl_research_favorites.py`; each season the best rule on earlier seasons was bet in the next, which hit 56.6% over 2011-2026). The rule it settled on covered **65.0% (13-7) on 1999-2005, 59.6% on 2006-2017 and 56.0% on 2018-2026** (+9.4% ROI at the real closing odds since 2015, the seasons after it was first chosen). The correction is more informative at night in every era; by day the favorites it favors cover about 50% at any size of lean. **Expect about 55%**: it came out of a scan and its evidence is thinner than the underdog rule's (p about 0.09 against break-even on the seasons after selection). A favorite pick needs a lean to the favorite, so it never coincides with an underdog pick.
- **Teaser legs:** the published Wong rule (2001): underdogs of +1.5/+2.5 teased 6 points to +7.5/+8.5, favorites of -7.5/-8.5 teased to -1.5/-2.5. Legs won **75.7% (underdogs) and 74.9% (favorites) since 1999**, weeks 5-16; since 2018 underdog legs 76.1% but favorite legs only 70.8%. A 2-team teaser needs 72.4% per leg at -110 and 73.9% at -120 and wins about 55-58% as a single bet. The value depends on the book's teaser price.
- **Wind unders (a separate game-day script, `python nfl_wind_unders.py`; section 12):** the under in open-air games with a forecast of 15+ mph over the game window, week 5 on. Unders went 57% on recorded wind of 15+ mph in every era since 1999 (499 games, weeks 5-16) and 56% on archived forecasts since 2018 (50 games); 10-15 mph forecasts went 60% (169 games) and are tracked as a watch list. The market lowers totals for wind by about half of what scoring falls.
- **What was overturned:** the previous rule (underdog on a 3/6/7/10/14 line with correction >= 0.25) showed 65.8% on 2006-2026 but went **53-50 (51.5%) on the untouched seasons**. It was a selection artifact (found by scanning about 25 conditions on the same data), and the half-point-line pattern even reversed there.
- **New data does not lift the ceiling:** opening lines and line movement (2007-2021), the starting quarterback's own rating, record and luck, pass rush and coverage, team popularity and retuned rating memory were each tested against a rule fixed in advance; none added information beyond the closing line, and none produced a favorite-side edge (section 6).

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

Ridge regression (alpha 300) of the home margin on ratings for **yards, turnovers, first downs, points and plays** (offense and defense), a margin Elo, rest-day difference, neutral site and quarterback features. Trains on the last 12 seasons; projections average `BOOTSTRAP_RUNS` (10,000) bootstrap fits, each the same ridge refitted on a resample of the training games drawn with replacement (Mean = their average, StdDev = their spread; `nm.bootstrap_projection` does this in numpy, matching the scikit-learn fit to 1e-13 points, in about 70 seconds). Margin error (2014-2026): 10.19 points (the closing spread 9.94). **It does not beat the spread** (about 51% against it). It is only a logged reference and the fallback fair line.

### 3.4 Model B: the correction model (`nfl_signal.py`, the thing the picks come from)

Ridge regression, **alpha 10000**, on the cover margin (`resid = result - spread_line`; positive = home covered), with features `PBP_FEATURES + BOX_FEATURES + BASIC_CONTEXT` (the 31 play-by-play and 5 box-score pairs, offense and defense, plus `neutral`, `rest_diff`, `elo_diff`), standardized, trained on the **14 seasons** before the target season. Its output is the **correction**: points by which it thinks the spread is off, positive meaning the home team covers. Strongly regularized, so most corrections are small (|correction| above 0.5 is uncommon). Production code: `week_corrections(season, week)`. `scored_history(table, season)` scores every season out of sample: 2006 on walk-forward, 1999-2005 by a model fitted on 2006-2019.

### 3.5 The rules (`nfl_signal.is_pick`, `nfl_signal.is_favorite_pick`, `nfl_signal.teaser_leg`, mirrored in the sheet formulas)

- **Pick:** `5 <= week <= 16` and `toward_underdog > 0` (and `>= SIGNAL_THRESHOLD`, default 0; the "Minimum lean to underdog (pts)" setting on the workbook's Model Info tab overrides it in the sheet). Underdog = the side getting points at the spread in the sheet; `toward_underdog` = the correction signed toward that side. Favorites are picked straight only by the favorite rule below.
- **Favorite pick:** `5 <= week <= 16`, a night game (`nfl_signal.is_night_game`: nflverse `gametime` of 19:00 Eastern or later; a Monday game without a kickoff time, as in the 1999 schedule, counts as night) and `toward_underdog <= -FAVORITE_MIN_LEAN` (0.5; the "Minimum lean to favorite (pts)" setting overrides it in the sheet). It needs a lean to the favorite, so it can never fall on a game the Pick rule takes.
- **Teaser leg:** `5 <= week <= 16` and the spread is 1.5/2.5 (the underdog, teased to +7.5/+8.5) or 7.5/8.5 (the favorite, teased to -1.5/-2.5). No model involved.

### 3.6 Weekly runner (`nfl_prod_projections.py`)

1. Load/refresh nflverse data, pick the next unplayed week (or `PREDICT_YEAR` / `PREDICT_WEEK` overrides in the `__main__` block).
2. Model A projections (logged to `C:/Users/Bobby/nfl_predictions_<timestamp>.txt` and the console). The whole run is logged in 8 numbered, timed steps, with bootstrap progress every 10%; a full run takes about 2 minutes.
3. Build the full research table, compute the hit-rate table (`backtest_hit_rates`, about 5 s), the favorite-pick record by era (`favorite_backtest`, about 5 s) and the teaser-leg rates, this week's corrections and reasons (`nfl_reasons.season_reasons`, one fit per season), and corrections, reasons and kickoff times for earlier weeks already in the sheet (backfill; never adds games).
4. Fetch DraftKings spreads (fallback: DraftKings NJ, DK PA, consensus, other books, opener; the book used is logged; a game that has kicked off gets its closing line, since the feed then also carries live in-game lines, which are skipped by their `is_live` flag), log the week's picks with their reasons, then the favorite picks (team, line, kickoff, lean and reason), and the teaser legs.
5. Append a timestamped snapshot of lines and betting splits to `data_cache/line_log.csv`.
6. Write/refresh the Excel workbook.

### 3.7 The Excel workbook (`nfl_sheet.py`)

One workbook per season, rebuilt in full every run after the games already in it are read back **by header text**, so nothing typed in is lost and older layouts convert in place (the pre-conversion file is kept once as `<year>_nfl_projections_before_new_layout.xlsx`; a file with no readable games sheet is copied to `*_unreadable_<timestamp>.xlsx`). It is saved to a temporary file and swapped in, so a failed save (for example with the file open in Excel) never leaves a half-written workbook; the runner logs the error. Three tabs:
- **This Week** (opens first): title and update time; a scoreboard (picks W-L-P and win %, the previous week's picks, teaser legs W-L and win %, average line versus the close) with a second, lilac row for the favorite picks (W-L-P, win %, the previous week's, all straight picks together and their win %, the favorite picks' average line versus the close); **this week's picks**, sorted by lean, each as a row (game, pick, lean, Hist. Win %, QB note, result) followed by a bold "Why:" headline and the reason's detail lines; the **favorite picks** in the same form (lean to the favorite, the rule's win % since 1999, kickoff and QB note); the **teaser legs** (leg, type, win % since 1999 and since 2018, result); every game of the week (spread, the side the model leans to and by how much, pick or favorite pick, teaser leg, result); a short legend. Six columns, about 800 px wide. Its cells are formulas into the Season Log (the reason text and lean text are written as values).
- **Season Log**: one row per game, fixed row height. Visible: Week, Away, Home, **Home Spread** (manual/yellow; negative = home favored), **Pick** (light blue, e.g. "Bills +3"), **Favorite Pick** (lilac, e.g. "Cowboys -4.5"), **Teaser Leg** (light orange, e.g. "Saints +8.5"), Lean to Underdog (pts; negative = toward the favorite), Hist. Win %, **Final Margin (Home)** (manual/yellow), Pick Result, Fav Pick Result, Teaser Result, Line vs Close (underdog picks), QB Note, **Notes** (manual/yellow), **Why the Model Leans** (the full reason; the cell is top-aligned so the row shows the headline). Grouped and collapsed at the right: Fair Line (Home), Fair Line vs Spread, Model Side, Covering Team, Model Side Result, Closing Spread (Home), Model Correction (Home), Kickoff (ET) (refreshed every run, since the league can flex a game into or out of a night slot), Fav Line vs Close, Year. Panes frozen at D2, header filters, week banding and outlines, win/loss shading.
- **Model Info**: the settings in B5:B8 (minimum lean to the underdog, first and last pick week, minimum lean to the favorite for favorite picks; read back by label on the next run, and the Season Log formulas reference them as `'Model Info'!$B$5` etc.), the hit-rate table at A13:C17 (lean from, win %, games; 1999 to date, weeks 5-16; Hist. Win % is a `LOOKUP` into it), the teaser table from row 22, the favorite-pick table (win %, games by era) from row 28, and a guide.

The reason (`nfl_reasons.py`) decomposes the correction exactly (ridge: intercept plus one contribution per rating), merges ratings that carry the same numbers or words, and groups the contributions into themes (special teams, pace, penalties, giveaways/takeaways, sacks taken/pass rush, passing/pass defense, running game/run defense, play calling, offense/defense, rest, Elo). Headline: the team the model leans to, by how much, the top three themes for it and the top one against (each at least 0.05 points), with "(worse)" when the team is worse on the theme's main rating yet it helps them, or "(better)" when it is better yet it hurts them (the line has overrated that gap). Detail lines: each headline theme's total effect and its main rating with both teams' numbers, then whether the effects are steady (same coefficient sign when fitted on the earlier and the later half of the training seasons). Reasons are stored once like the corrections, so the sheet shows what was known before the game.

Behaviors worth knowing:
- **Closing Spread** is filled from nflverse once a game has been played; **Line vs Close** is how many more points the pick got than the close (positive = beat the close), the fastest live check of whether the picks carry an edge.
- **Fair Line = -spread + correction**, in half points, never rounding the correction to zero, so the Model Side always agrees with Model B. The Model Side on a non-Pick is not a bet.
- A spread already in the sheet is **never overwritten**; for unplayed games the run logs a warning when DraftKings' current line differs (a pick or teaser leg holds only at the line you can bet). A week written after some of its games have kicked off gets those games' closing lines; before October 2026 the fetch could pick up a live in-game line for a game under way (finished games were unaffected).
- Formulas were checked with a small Python evaluator of Excel's semantics (October 2026): a full replay of 2025 through the writer gave picks 34-29-1 and teaser legs 40-20 from the workbook's own formulas, the same as a direct count from the data and the research harness, with no pick decision differing from `nfl_signal.is_pick`. The favorite-pick change (October 20, 2026) was checked with pycel, run on the same inputs through the previous writer (git HEAD) and the new one on a copy of the live 2026 workbook: every pre-existing Season Log column evaluated identically (43 games x 23 columns), the This Week picks and teaser sections were identical, and Favorite Pick, Fav Pick Result and Fav Line vs Close agreed with `nfl_signal.is_favorite_pick` and the results on every row (pycel lacks `REPT`, supplied as a plugin, and its `TRIM` keeps leading spaces, which Excel's drops). **The workbook has not been opened in Excel in this environment.**
- A favorite pick needs a kickoff time on its row; rows written before October 20, 2026 get theirs on the next run (current week and backfilled weeks).

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
| **Night-game favorites the correction favors by 0.5+ (current Favorite Pick rule)** | 3-12 | **65.0%** (n=20) | 59.6% (n=136) | 56.0% (n=100) |
| Day-game favorites the correction favors by 0.5+ | 45-52 | 53.2% | 50.3% | 48.4% |
| No model: every favorite | 150-170 | 49.2% | 49.8% | 48.7% |

Information of the correction (all games, weeks 5-16): correlation with the cover margin 0.084 / 0.056 / 0.064, regression slope t = +2.7 / +2.5 / +2.3. Hit rate does **not** rise reliably with the size of the correction (underdog side by size bins: 53-59% in most bins; the largest corrections, >= 1.0, went 14-29 on 1999-2005).

Selection-process test: re-running the "scan 78 rules, keep the best on earlier seasons" process walk-forward on 2006-2026 picks the key-number rule every season from 2011 and keeps hitting 64% there, so the rule was stable within 2006-2026 but did not carry to 1999-2005. Empirical-Bayes shrinkage across the scanned conditions put it at about 60% even before the untouched test.

Favorites (October 20, 2026; `nfl_research_favorites.py`, pre-registered in its docstring before each run): the same selection process run on favorites (26 conditions x 3 thresholds on the lean to the favorite) hit **56.6%** over 2011-2026 (55.3% with the stricter lower-bound selector) and settled on night-game favorites with a lean of 0.5+. Unlike the key-number rule it held on 1999-2005: 13-4 as scanned, 13-7 once the 1999 Monday night games are counted (the 1999 schedule has no kickoff times), though that is only 20 games. For it: the correction x night interaction is positive in every era, for the cover margin and for covering (pooled t = 1.74); with \|correction\| >= 0.5 the model's side covered 69% / 58% / 58% at night against 52% / 52% / 51% by day, on both sides; the rule holds on Thursday, Sunday and Monday nights and for home and road favorites; 77-59 (56.6%) in the seasons after the process first chose it. Against it: it came out of a scan, weeks 1-4 (outside the pick weeks) went 42-44, and the seasons after selection give p about 0.09 against break-even. Reading: standalone national-TV games draw the most recreational money, so their lines stray furthest from the play-by-play fundamentals; that clears the usual favorite tax at night but not by day.

Reproduce: `python nfl_research_validate.py` (untouched test, process test, shrinkage, information by era, side and size, robust rule, the 60% requirement), `... models` (production versus the 4-stat model), `... threshold` (both-sides threshold rules with thresholds set from earlier seasons), `... teasers`; `python nfl_research_favorites.py` (spread-total scaling and two classic favorite angles), `... scan` (the favorite selection process), `... teasers` (favorite teaser legs and the total), `... night` (robustness of the night-game rule).

## 5. What was found along the way

1. **Model A has no edge against the spread** (MAE 10.19 vs the market's 9.94). Re-tested October 20, 2026 with confidence scores (`nfl_research_margin.py`, all eras): its 10,000-run average differs from a single fit by 0.08 points, and the spread of the runs (about 1 point) is the model's wobble, not the game's (13.4 points), so "high confidence" games covered 49-53%; taken at its word it called games 60-65% that covered 51%.
2. **The correction model carries a small, stable signal** (section 4). Its underdog picks beat the no-model underdog baseline by about 3-5 points in every era; its favorite picks do not (about 51%). The underdog/favorite asymmetry held on the untouched seasons too (53.7% vs 51.2%), consistent with the market shading lines toward favorites.
3. **The key-number concentration was noise.** It looked strong in 2006-2026 (and survived a nested selection test there) but disappeared on 1999-2005, where half-point lines did best instead.
4. **The size of the achievable edge is set by the signal's information.** In a bivariate-normal picture, a signal correlated rho with the cover margin, betting its side on the games where it is strongest, gives: rho 0.07 -> about 56% on 12 picks a season, 55% on 24-36. 60% needs rho about 0.11 (12 picks), 0.13 (24), 0.14 (36). Every honest rule tested landed where this predicts.
5. **A 4-stat model is more informative but not more profitable.** Forward-selected on 2006-2017 only: offense drives per game, special-teams EPA, rushing success rate, interception rate. Pre-registered test on the two eras its selection never saw: correlation 0.102 vs 0.084 (1999-2005) and 0.110 vs 0.064 (2018-2026); binary correlation 0.065 vs 0.036 and 0.063 vs 0.032. But with thresholds set from earlier seasons its both-sides picks hit 54.5% pooled (60.3% / 53.2% / 54.0% at about 2 a week), the same band as the production rule, so production was not switched. Its coefficients read as market blind spots: more drives per game -> covers less (the market overrates per-game volume), better special teams -> covers more, better rushing success and more interceptions forced -> covers more.
6. **What wins games is not what predicts them, and the market knows both** (`nfl_research_winning.py`, October 20, 2026). In the game itself, per-play efficiency and turnovers decide it (EPA per play in competitive game states r = 0.55 with winning, giveaways -0.39, takeaways +0.38; net yards only 0.29). Before kickoff, offensive efficiency predicts the winner best (points per game 0.31, EPA per play 0.30), defense much less (best -0.21), and turnovers barely (0.13-0.17), because offense repeats and turnovers and most defensive numbers do not (split-half reliability: EPA per play 0.62 offense vs 0.39 defense; fumbles lost 0.07; takeaways 0.27). The closing spread (0.38) beats every stat and every stat combination out of sample. The market prices passing, points, yards, first downs and turnovers within about half a point per SD of their worth; it underprices rushing efficiency (worth about 3.3 points per SD, priced 2.5) and early-down success, overprices pace, and overweights stats that do not repeat (mostly defense relative to offense). Those errors are what the correction model already bets: a stability composite and the four stats that passed the strict test add little or nothing to it.
7. **Totals: the market under-prices wind** (`nfl_research_totals.py`, October 20, 2026; the first totals research in the project, so every era was clean). A totals correction built like the spread one (rating sums plus weather and context) failed: no information on 1999-2005, and the closing total is close to unbiased (unders 49.7% / 50.0% / 51.5%). But the pre-registered wind angle passed in every era: outdoor games with 15+ mph of recorded wind went under 57.1% / 55.9% / 58.5% (499 games, about 18 a season, about +10% ROI at the real odds). The under rises steadily with the wind (47% calm, 50% at 5-10 mph, 55% at 10-15, 57% at 15-20), the market lowers the total by only about half of what scoring actually falls (15-20 mph: 1.5 points vs 2.9), it is wind and not cold (cold but calm games go over), and it held on archived weather forecasts from 2018 on (Open-Meteo; 15+ mph forecast 28-22, 56%; 10-15 mph forecast 60% on 169 games), so it is not a weather-surprise artifact. Not in production; the next step is a game-day forecast check, tracked forward.

## 6. What was tried and rejected (do not redo without a new idea)

- **The owner's gut stats** (yards, first downs, turnovers, plus points/plays): no information beyond the spread in weeks 5-16.
- **Key-number rule** (underdogs on 3/6/7/10/14 lines): failed the untouched test (51.5%).
- **Quarterback-change filter** (skip games where either team's starter differs from its previous game; the team ratings go stale): helped on 1999-2005 (45.7% for underdogs whose starter changed), hurt on 2006-2017 and 2018-2026 (those games hit 56-75%). Not adopted.
- **Size thresholds and top-N-per-week rules** on either model: 52-56% wherever tested out of sample; no threshold concentrates the edge reliably.
- **Training window** (re-tested October 2026 against all three eras, `python nfl_research_validate.py window`): fitting on only the last 3, 5, 8 or 10 seasons is worse than 14, most of all in the recent era where era drift should help it (correlation with the cover margin in 2018-2026: 3 seasons 0.015, 5 seasons 0.026, 8 seasons 0.042, 14 seasons 0.064; underdog picks 52.3% with 5 seasons vs 55.0% with 14). The projection model agrees (average miss 9.985 points with 5 seasons vs 9.967 with 12). Era changes are absorbed by the team ratings, which are rebuilt from each team's own recent games; the regression only learns how ratings translate into margins and covers, which is stable, and the cover signal is so small that it needs many seasons to estimate.
- **Weighting a team's last 3-4 games more heavily** (`python nfl_research_recency.py`, October 20, 2026; pre-registered). The ratings already weight recent games (decay 0.90 per game: the last four carry about 30% of a mid-season rating), and decay 0.80 had been flat in the rating-memory grid (`nfl_research_data.py`). Sharper recency makes things worse, steadily: the correction's correlation with the cover margin falls from 0.083 / 0.055 / 0.064 to 0.074 / 0.052 / 0.050 at decay 0.75 (last four games about 54%) and 0.071 / 0.042 / 0.042 at decay 0.60 (about 67%), with the underdog picks down to 53.7% and 51.7% in 2018-2026; recent-form ratings added on top of the current ones add nothing (0.084 / 0.051 / 0.061); the projection model's error barely moves (within 0.03 points) and its gap loses information. Recent games are mostly noise on top of a team's longer record, and the market already leans on them.
- **Skipping very bad underdogs** (the owner's "don't back pathetic teams", `python nfl_research_validate.py bad`): the model's underdog picks on bottom-15% Elo teams covered 50.0% / 57.1% / 57.0% across the eras against 54.8% / 54.8% / 54.4% for its other underdog picks (55.3% vs 54.7% overall), and on teams with a .250-or-worse record 55.8% / 56.8% / 55.9%; bad underdogs the model did not pick covered 50.3%. The market already prices how bad a team is, so excluding them would remove picks that did as well or better.
- **Sign-stable features** (`python nfl_research_validate.py stable`; inside each fit, keep only features whose effect points the same way in both halves of the training seasons): a tie with production (0.088 / 0.055 / 0.066 vs 0.084 / 0.056 / 0.064), so production stays. A related diagnostic (`... components`) found the offense ratings carry steady information in every era (0.063 / 0.063 / 0.052) while the defense ratings' share swings (0.090 / 0.022 / 0.063): the "bad recent defensive numbers regress" effect has flipped direction across eras (negative before 2013, strongly positive in 2013-2019, near zero since), whereas pace (fewer drives per game covers more) and rushing success (better covers more) have the same direction in every era.
- **Agreement filters, consensus and averaging of models**: no gain (the production + 4-stat average landed between them).
- **Underdog-frame models, favorite-side models trained on subsets**: 48-58%, within noise; favorites 48-54%.
- **Other features/models**: market prices (moneyline-implied, juice lean), weather/context, cover history, overreaction, travel, ticket/money splits (2023+ only), QB injury flag and injury counts, gradient boosting, logistic classification, a denoised two-stage target: none added reliable information.
- **Opening-line betting** (apparent 61%): an artifact of stale Action Network openers.
- **Monte Carlo** (`nfl_sim.py`, `nfl_research_sim.py`): drive-based scoring simulator reproducing key numbers; simulated-EV selection did not beat simple rules; moneyline underdogs no edge; Kelly: flat 1% about x1.35 over 15 seasons (18% max drawdown), half Kelly about x2 (46% drawdown).

**New data, October 2026** (`nfl_research_data.py`; each block was added to the production correction model and judged by a rule fixed before running: it must raise the correction's correlation with the cover margin in every era where the data exists; favorites were scored separately throughout). None passed:

| Data (source, years) | What it measures | Result |
|---|---|---|
| Opening and closing lines (Sportsbook Reviews Online archive, 2007-2021, `nfl_market_history.py`; lines move 1.4 points on average) | Does the model see what the market learns during the week? | The market moved toward the model's side only in 2015-2021 (correlation +0.105, about +0.16 points a game), not in 2007-2014 (+0.013). Picks graded at the open hit the same as at the close (underdogs 56.0% vs 55.9%, favorites 52.2% vs 52.7%); deciding and betting at the open: underdogs 57.3% vs 56.1% in 2007-2014 but 54.4% vs 55.7% in 2015-2021. A model trained against the open was no better (0.066 vs 0.070). Open-to-close movement does not predict covers at the close (0.008). |
| Starting quarterback's own rating (nflverse player stats 1999+, EPA per play over his career) | Ratings that follow the actual starter | Predicts the final margin strongly (correlation 0.33) but the cover margin barely (0.04): the market prices quarterbacks. Model information unchanged (0.084 / 0.057 / 0.064 vs 0.084 / 0.056 / 0.064). |
| Record and luck (schedules 1999+: win %, win % minus Pythagorean, one-score record) | What the public sees | Slightly worse in all three eras. |
| Pass rush and coverage (QB hits per dropback 2006+, passes defended 1999+, opponent-adjusted) | Process stats behind sacks and interceptions | Slightly worse (2010-2017 0.050 vs 0.053; 2018-2026 0.061 vs 0.064). |
| Popularity (primetime games the league schedules, 1999+) | Books shading lines against public teams; would make favorites playable when the underdog is the popular team | Wrong direction: popular teams covered as often or more (correlation with the cover margin -0.004 / +0.014 / +0.061); betting against them went 41-47% in 2018-2026. |
| Rating memory (9 settings of per-game decay and season carry) | Ratings tuned for the market's errors rather than for the margin | Flat (0.045-0.087); the best on 2006-2017 is effectively the current setting. |

What this says: every public signal tested is already in the closing line. The 55% ceiling is about public data against an efficient market, not about which public data this model happens to use. **Favorites:** across the production model and every block above, the favorite side of the correction covers 50-54% (production: 51.2% / 52.7% / 50.7% across the eras), so no straight-bet favorite rule is supported by day; the exception found in October 2026 is night games (section 4), and the one other favorite angle with long support is the Wong favorite teaser leg (74.9% since 1999, 70.8% since 2018).

**Favorites, October 20, 2026** (`nfl_research_favorites.py`; pre-registered, each judged on all three eras):

| Idea | Why it might work | Result |
|---|---|---|
| Spreads scaled to the scoring environment: \|spread\| x (total / league total - 1) | Spreads come from additive power ratings, but margins grow with scoring, so favorites in high-total games would be under-laid | No information: correlation with the favorite's cover margin -0.009 / -0.014 / +0.024 (pooled -0.002, n=4,769). With the model's lean (Rule A): 50.0% / 51.2% / 50.4%. The market already scales spreads with the total. |
| Divisional compression (fade divisional favorites) | Rivals meet twice a year, which compresses margins | +0.035 / -0.004 / +0.016: no. |
| Favorite-longshot (big favorites overbet) | The public over-bets big favorites | -0.018 / -0.007 / -0.024 (bigger favorites covered no less): no. |
| Favorite teaser legs only when the total is under 49 (the published Wong refinement) | A teaser costs the same in every game, but margins spread out as scoring rises | Legs with a total under 49 won 78.0% / 81.6% / 71.4%, the last below break-even; across all favorites the teased cover rate does not fall as the total rises. Not adopted. |
| Selection process over 78 favorite rules | A disciplined scan, bet only on seasons it never saw | Passed: 56.6% over 2011-2026 and 13-4 on 1999-2005, settling on night-game favorites with a lean of 0.5+. **Adopted** as the Favorite Pick rule (section 4). |

**Team stats and winning, October 20, 2026** (`nfl_research_winning.py`; pre-registered, judged on all three eras; nothing in production changed):

| Test | Why it might work | Result |
|---|---|---|
| M1: each of the 74 ratings against the cover margin (same sign in every era, Bonferroni \|t\| >= 3.4, adds to the correction in every era) | A stat the market misprices | 4 pass: EPA per run, rushing success rate, early-down success rate (all offense, +0.05 to +0.08) and opponent drives per game (-0.05). But their partial correlations given the correction are 0.00-0.05: the correction already bets them (the blind spots in item 5 of section 5). Forward watching only. |
| M2: stability vs mispricing across stats | The market overreacts to stats that do not repeat | Passes (Spearman +0.51 / +0.49 / +0.49), but mostly as offense vs defense: within one side it is weak or mixed. |
| M3: composite of the most reliable minus the least reliable third of the ratings | Bet the stable stats, fade the noisy ones | Tracks the cover margin in every era (+0.04 / +0.07 / +0.04) but adds nothing to the correction (partial -0.02 / +0.04 / -0.00, t = 0.9). |
| M4: last week's turnover margin | The market overreacts to turnover luck | No: -0.017 / -0.004 / +0.007 (t = -0.3); the market discounts it. |

**Margin models with confidence scores, October 20, 2026** (`nfl_research_margin.py`; pre-registered; weeks 5-16, all three eras): predict the final margin, attach a confidence, bet the side the projection favors. Rule needed: honest bets at 55%+ hitting 55%+ and information beyond the correction in every era. All three failed.

| Margin model | Error vs the line (1999-2005 / 2006-2017 / 2018-2026) | Gap information | Confidence |
|---|---|---|---|
| A: Model A refitted weekly with bootstrap runs (the original method) | 10.13 / 10.79 / 10.06 vs 10.01 / 10.59 / 9.83 | +0.048 / +0.011 / +0.017 | Gap buckets 49-52% under 7 points; bootstrap z 49-53%; "60-65%" covered 51%; honest bets at 55%+ went 18-19 |
| K: Kalman-filter team strengths with their own uncertainty | 10.09 / 10.75 / 10.16 | +0.079 / +0.013 / +0.004 | Stated SD 13.7-14.1 points for 90% of games, so confidence hardly varies; "60-65%" covered 52%; honest bets at 55%+ went 4-3 |
| F: every rating aimed at the margin | 10.06 / 10.76 / 10.06 | +0.091 / +0.016 / +0.030 (-0.011 in 2006-2017 given the correction) | Gap buckets 52-55%; "60-65%" covered 51%; honest bets at 55%+ went 19-21 |

The usable form of "margin plus confidence" is the one in production: spread plus correction (as accurate as the line, information +0.083 / +0.055 / +0.063), with Hist. Win % as its calibrated confidence.

**Totals, October 20, 2026** (`nfl_research_totals.py`; pre-registered; weeks 5-16; eras 1999-2005 / 2006-2017 / 2018-2026):

| Test | Result |
|---|---|
| Totals correction (ridge on both teams' rating sums plus dome, wind, temperature, night, divisional, neutral, rest), correlation with the over margin > 0 in every era | Fail: -0.004 / +0.050 / +0.084. The rating sums alone carry nothing on 1999-2005; the weather and context part carries what there is (+0.063 / +0.061 / +0.053, a diagnostic). |
| Every under the correction favors (the underdog-rule analog) | Fail: 47.2% / 50.7% / 58.1% (the model leaned over almost everywhere before 2018). |
| Unders with 15+ mph of recorded wind outdoors | **Pass**: 57.1% / 55.9% / 58.5%, 499 games, about 18 a season. |
| The same on archived forecasts (Open-Meteo, 2018 on, wind averaged over the game window) | **Pass**: 28-22 (56.0%); 10-15 mph forecasts went 60.4% (169 games). |
| Unders in night games, divisional games, totals 10% above the league average | Fail (night 52.3% / 49.5% / 54.8%; divisional 50.4% / 54.0% / 50.4%; high totals 51.1% / 53.8% / 55.1%). |

**Not tried:** Bayesian hierarchical models, Poisson/Dixon-Coles team models, neural nets/GPs, injuries weighted by snap share (snap counts 2012+), short-history data (Next Gen Stats 2016-2024, participation/personnel 2016-2025, PFR pressures 2018+, FTN charting 2022+), totals-market angles beyond the spread-total scaling above, combining the model with teaser legs, real-time news/line speed (needs `line_log.csv` to accumulate). **Why short-history data cannot settle the question:** telling a 0.11 correlation from today's ~0.065 takes on the order of 2,000 test games (about 11 seasons of weeks 5-16) even before allowing for the selection of the idea; data that starts in 2016 or 2018 leaves 4-8 test seasons after training, so even a real improvement could not be confirmed from history, only forward over years.

## 7. Weak points a fresh reviewer should interrogate

1. **Forking paths are mostly spent.** 2006-2026 has been examined in hundreds of ways; 1999-2005 has now been used once for validation (and a handful of diagnostics). The next clean evidence is forward: 2026 weeks 5-16 onward. Keep any new idea pre-registered and judged on information (correlation), which has far more statistical power than hit rates on small pick sets.
2. **Closing line versus bet time.** Backtests bet the closing spread; the owner bets earlier in the week. Whether the picks beat the close (closing-line value) is the fastest real-world test of an edge; the workbook now measures it per pick (Line vs Close) and on average (the This Week scoreboard), from 2026 week 5 on, with `line_log.csv` recording how lines move during the week.
3. **Lines are the nflverse consensus close**, not DraftKings; DraftKings lines, juice and limits differ. Teaser value depends entirely on the book's teaser price (-120 or better needed; -130 removes the edge).
4. **Favorite teaser legs have weakened** (70.8% since 2018, below break-even); underdog legs have not. That split was seen after the fact, so it is noted rather than acted on.
5. **The 60% target.** No rule tested reaches it out of sample; the measured information supports about 55%. Volume at 55% is plentiful (4-5 a week), which matters more for profit than a higher rate on fewer picks.
6. **Workbook not yet opened in Excel**; its formulas were checked only with Python evaluators, and the Pick, Favorite Pick and Teaser formulas re-implement `nfl_signal.is_pick` / `is_favorite_pick` / `teaser_leg` and could drift from them (the 2025 replay and the pycel comparison in section 3.7 are the checks to rerun after changing any of them).
7. **The favorite rule came out of a scan.** The process that found it was honest (each season's choice used only earlier seasons) and it passed on 1999-2005, but that is 20 games, the night x correction interaction is only t = 1.74 pooled, and "night" is partly a proxy (more Thursday games since 2012, flexed Sunday nights). Expect about 55% and judge it forward on its record and on Fav Line vs Close. To switch it off, set Minimum lean to favorite to 99 on Model Info (the workbook) and `FAVORITE_MIN_LEAN` in `nfl_signal.py` (the run log); nothing else depends on it.

## 8. Repository map (`nfl/`)

Production: `run_all.bat` and `nfl_run_all.py` (one click for all three weekly flows, section 13), `nfl_prod_projections.py` (runner), **`nfl_wind_unders.py`** (the separate game-day wind check for totals, section 12), `nfl_model.py` (Model A), `nfl_signal.py` (Model B, the pick, favorite-pick and teaser rules, the backtest tables), **`nfl_reasons.py`** (the plain-words reason for each game's lean: exact decomposition of Model B's correction, themes, both teams' numbers, better/worse, steadiness; `season_reasons(table, season, weeks)` is what the runner calls), `nfl_sheet.py` (the three-tab workbook and its formulas, DraftKings spreads, QB flags), `nflverse_data.py`, `nfl_pbp_features.py`, `nfl_injury_features.py`, `nfl_lab.py` (rating/feature table and walk-forward harness), `nfl_market_moves.py` (Action Network lines/splits and the line log), **`nfl_explain.py`** (the full breakdown of any pick: `python nfl_explain.py TEN` lists the dozen ratings that pushed the correction most toward or away from a team, with both teams' numbers and whether the team is better or worse there, marks each effect steady or flipping between the earlier and later halves of the training seasons, and prints the reason the workbook shows).

Research (each runnable, prints its evidence): **`nfl_research_validate.py`** (the untouched-data validation, model comparison, threshold rules, teasers; start here), **`nfl_research_favorites.py`** (favorites: spread-total scaling, divisional and favorite-longshot angles, the favorite selection process that found the night-game rule, favorite teaser legs and the total, and the night-game robustness checks), **`nfl_research_winning.py`** (which offense and defense stats go with winning, which repeat, which predict the winner before kickoff, and which the market misprices), **`nfl_research_margin.py`** (margin models with confidence scores: the original bootstrap method, a Kalman-filter rating model and an all-ratings model, against the spread), **`nfl_research_recency.py`** (weighting each team's last 3-4 games more heavily: the pick model, recent form on top of the current ratings, and the projection model), **`nfl_research_totals.py`** (totals: a totals correction, classic over/under angles, and the wind under on recorded weather and on archived forecasts), **`nfl_research_data.py`** (the new-data tests: opening lines, bet timing, starting-QB ratings, record and luck, pass rush and coverage, popularity, rating memory; `nfl_market_history.py` loads the 2007-2021 opening and closing lines), `nfl_research_select.py` (the condition scan that produced the now-rejected key-number rule), `nfl_research_features.py` (stat sets, single stats, forward selection, frontier, underdog frame, consensus), `nfl_research_weekly.py`, `nfl_research_window.py`, `nfl_research_signal.py` (original underdog-signal evidence), `nfl_research_sim.py` and `nfl_sim.py` (Monte Carlo), `nfl_research_model.py`, `nfl_research_ats.py`, `nfl_research_injuries.py`, `nfl_lab_extra.py`, `nfl_prod_backtest.py`.

Legacy (Pro-Football-Reference scrapers, broken since the Cloudflare block): `nflpicks.py`, `nflpicks_backtest.py`, `nflpicks_v2_0.py`, `nflpicks_v2_0_multithreaded.py`.

Player props (section 11): **`nfl_prod_props.py`** (weekly runner; `SIDES` picks overs and/or unders), `nfl_prop_lines.py` (DraftKings/ESPN BET prop lines from ESPN: current week, cached history, opening game lines, snapshot log), `nfl_prop_model.py` (features, per-market forecasts and the early-week variant, outcome distributions, reasons), `nfl_prop_signal.py` (the under and over rules, Bet At prices, grading, backtest scoring), `nfl_props_sheet.py` (the props workbook), **`nfl_research_props.py`** (forecast quality, discovery, the untouched tests, the over research).

Key framework functions in `nfl_lab.py`: `default_table()` (cached full table, 1999 on), `build_table(pairs, refresh)`, `walk_forward(table, fit_predict, first_test=2006, last_test=2026, window=14)`, `ridge_margin(columns, alpha, target)` (drops near-constant columns, returns zeros if none remain), `information_test`, `scorecard`; constants `DISCOVERY_END = 2017`, `PBP_FEATURES`, `BOX_FEATURES`, `BASIC_CONTEXT`.

## 9. Environment and gotchas

- Windows; **32-bit Python 3.8.5**; pandas 2.0.3, numpy 1.24.4, scikit-learn 0.24.2, scipy 1.5.2, openpyxl, requests. Not installed: lightgbm, xgboost, statsmodels, torch, pymc.
- Run scripts from inside `nfl/`. `lab_table_default.pkl` caches the research table; delete it (or `default_table(refresh=True)`) after changing rating code.
- The workbook path comes from `nfl_sheet.get_desktop_path()` (OneDrive-aware); tests should monkeypatch it to a temp folder and copy the real sheet there (the writer also takes `output_dir`). Close the workbook in Excel before a run, or the save fails. The runner's log path (`C:/Users/Bobby/...`) is hardcoded. `BOOTSTRAP_RUNS` in `nfl_prod_projections.py` is 10,000 (the owner's choice, matching his earlier 10,000-run regression approach); pass `num_runs` to `main()` for quick test runs.

## 10. Suggested next steps

1. **Measure live performance properly**: done in the sheet (closing spread, line value versus the close, and the season scoreboard); track the picks from 2026 week 5 on, and the favorite picks from 2026 week 7 on (their own scoreboard row and Fav Line vs Close). Closing-line value tells within a season or two what win rate needs many seasons to show.
2. **New information has to come from outside the public box score.** Free public data has now been tested broadly (section 6) and is priced. What remains: information the market does not have yet (reacting to injury/inactive news before lines move; `line_log.csv` will show how fast lines react), markets that are less efficient than NFL sides (player props, now built: section 11; team totals, first halves, live betting, college football), short-history data that can only be judged by forward testing, and paid data (charting and grades). Judge any new signal by its correlation with the cover margin in every era, with about 0.11+ as the bar for a 60% rule.
3. **Teasers**: if the book prices 2-team 6-point teasers at -120 or better, pairing underdog legs is the best-supported bet in the project; whether the model can filter legs is untested and would need a pre-registered test.
4. **Wind unders**: built as `nfl_wind_unders.py` (section 12); track its season log forward, including the 10-15 mph watch list, before trusting it with full stakes.

## 11. Player props (`nfl_prod_props.py`, October 2026)

A separate weekly flow that finds DraftKings player prop bets. It leaves the spread runner, its models and its workbook untouched (the only shared-code change is two additive loaders in `nflverse_data.py`: `load_players()` and a `columns` filter on `load_player_weeks()`).

### 11.1 What runs

`python nfl_prod_props.py` (from `nfl/`; about a minute, or 5-6 minutes the first time while it builds the forecast caches and downloads the prop-line history). `PREDICT_YEAR` / `PREDICT_WEEK` in the `__main__` block override the season and week, as in the spread runner. **`SIDES`** picks the rules: `('Over',)` by default (the owner prefers overs), `('Over', 'Under')` adds the under picks. Two rules, each with its own model variant (section 11.4 for the unders, 11.7 for the overs):
- **Overs: run it as soon as DraftKings posts the week's props (usually Tuesday or Wednesday) and bet right away.** An over is a pick only while its line is still at the opening number, because the edge is the market moving toward these overs during the week. An over still at its opener within 48 hours of kickoff is marked LATE and priced as the weaker bet it is.
- **Unders** hold against closing lines, so they suit bets placed close to kickoff.

Seven logged steps:
1. Load nflverse player stats (2006 on, offense only), the schedule, injury reports and the players table (ESPN ids); pick the next unplayed week (`find_next_week_to_predict`, reused from the spread runner).
2. Fetch the week's DraftKings prop lines from ESPN for every game **not yet kicked off** (after kickoff the feed carries in-game lines).
3. Build pre-game features and fit one model per market on the 10 seasons before this one: the early-week variant for the overs (no injury report, opening spreads and totals from ESPN), the full model for the unders (the week's injury report, closing game lines). Players are placed on their team by nflverse's current roster (the team of a player's last game is stale after a move). A quarterback DraftKings posts passing props for is used as his team's starter when nflverse lists someone else who has none (nflverse's expected starter can lag injury news).
4. Score every past prop line since 2024 (the backtest record behind Hist. Win %: overs at opening lines, unders at closing lines) and build the outcome distributions.
5. Forecast this week's players, apply the rules, log the picks best first with a one-line reason.
6. Grade earlier picks from nflverse results and append the week's lines to `data_cache/prop_line_log.csv`.
7. Write `<Desktop>\<year>_nfl_props.xlsx`; the log goes to `C:\Users\Bobby\nfl_props_<timestamp>.txt`.

**How to use the output:** each pick has a **Bet At** price, the worst DraftKings price at which the pick still returns 3% at its market's backtest win rate. The free feed has DraftKings' lines but **not its prices**, so check the price in the app and skip any pick priced worse than Bet At. Neither rule's win rate keeps rising reliably above its 0.65 threshold, so the order of a list is not a ranking of quality. If you cannot bet them all (about 30 overs a week early in the week; 70-80 unders), prefer the ones whose price beats Bet At by the most.

### 11.2 Data (`nfl_prop_lines.py`)

| Source | Content |
|---|---|
| ESPN core API `.../events/{id}/competitions/{id}/odds/{provider}/propBets` | The book's player props per game: the current line and the opening line, **no prices**. Provider 100 = DraftKings (late 2025 on), 58 = ESPN BET (2024-2025). Finished games stay in the feed. `games.csv` carries ESPN's event id (`espn`) for every game; nflverse `players.csv` maps ESPN athlete ids to gsis ids (99.8% of lines). |
| Markets used | Passing yards, completions, attempts, TDs, interceptions; rush attempts, rushing yards; receptions, receiving yards; pass + rush yards; rush + rec yards (`MARKETS`). Longest-play, kicking, defensive and touchdown-scorer props are skipped (no nflverse result or no line). |

Cached in `data_cache/props_espn_{season}.csv` (one fetch per finished game, about 40 seconds for two seasons): 19,909 main lines in 478 games of 2024-2025 (ESPN BET, plus DraftKings for 184 lines late in 2025) and 3,917 in 80 DraftKings games (2026 weeks 1-6). Two traps, both handled:
- **ESPN BET lists alternate lines** (a ladder such as 149.5/174.5/224.5/299.5 passing yards, each listed once); the main line is the one listed twice (its over and under). Players whose top count is tied between two values are skipped.
- **In-game lines.** DraftKings keeps moving a line after kickoff and the feed keeps the last value (a passing-yards line 258.5 -> 271.5 updated mid-game), so a stored DraftKings line counts as a closing line only if it was last updated before kickoff (`pregame`): 57-80% of receptions/attempts/TD lines, but only 5-27% of yardage lines. ESPN BET's stored lines are batch-stamped hours after each game, but where they moved they land only slightly nearer the result than the opener (receiving yards MAE 22.6 -> 21.8, passing yards 56.5 -> 54.9), as a pre-game close does and an in-game line would not, so they are treated as closing lines. Opening lines are always pre-game.

### 11.3 The forecast (`nfl_prop_model.py`)

One `HistGradientBoostingRegressor` (Poisson loss, 300 trees of 15 leaves, `min_samples_leaf` 100) per market, refitted each season on the 10 seasons before it, on the players books post such props for (the starting QB for passing markets; RBs, starting QBs and anyone averaging 2+ carries for rushing; RBs, WRs and TEs for receiving). Features, all known before kickoff:
- the player's recent (half-life 1.5 games) and longer (6 games) recency-weighted averages and his season-to-date average of attempts, completions, yards, TDs, interceptions, sacks, carries, targets, catches and air yards, and of his share of the team's targets, carries and air yards; games played;
- the team's implied points from the closing spread and total, against what it has been scoring; the spread; home field; a roof; the team's recent pass attempts and carries;
- the opponent's recent yards, catches and targets allowed (to the player's position group for receiving), sacks, points, each relative to the previous season's league average;
- the share of the team's targets and carries belonging to teammates ruled Out or Doubtful that week; whether the player is Questionable;
- the starting QB's recent passing, how many games he has started, and whether he is new.

**Chance over/under a line** (`OutcomeDistribution`): among the out-of-sample forecasts of the previous 6 seasons, take the games whose forecast was nearest (400 for quarterback markets, 1,200 rushing, 2,000 receiving), centred on this forecast; the chance is the share of their results above or below the line (for yardage markets each past result is scaled by the ratio of the forecasts). This doubles as calibration: a model that over-forecasts high-volume players is corrected by what those forecasts actually produced. Out-of-sample forecasts for every season from 2018 are cached in `data_cache/prop_forecasts_v{MODEL_VERSION}.csv` (bump `MODEL_VERSION` after changing features or settings).

Forecast quality (out of sample, 2018-2025, `python nfl_research_props.py quality`): better than the player's recency average alone in mean absolute error in nearly every market and season (passing yards 57-64 vs 63-68; receptions correlation 0.65-0.68 vs 0.63-0.67; rush attempts 0.75-0.81). Interceptions are close to unpredictable (correlation 0.0-0.16). The bias in passing yards swings with the league's passing environment (-3.8 to +9.1 yards a season).

### 11.4 The rule (`nfl_prop_signal.py`) and the evidence (`nfl_research_props.py`)

Protocol as in section 4: the rule was chosen on the **discovery** lines (ESPN BET 2024-2025) and then scored once on the **untouched test** lines (DraftKings 2026 weeks 1-6). Win rates are at the line (no prices; pushes and voids left out).

**Discovery findings.**
- Against **opening lines** the model looks strong (information 0.100, t 14; win rate rising to 61% at a 0.70+ chance). But the features use information that arrives during the week (the final injury report, the closing spread and total, the actual starter), and lines that rose 30% or more during the week ended over the opener 67% of the time. Against openers the edge is overstated.
- Against **closing lines**, the honest benchmark: information 0.044 (t 6.2), all of it on the **Under** side (unders 0.042, t 4.6; overs -0.005, about 50% at every threshold). The model's unders won 50.8% / 52.0% / 53.1% / 56.2% / 56.3% at chances 0.50 / 0.55 / 0.60 / 0.65 / 0.70+, against 51.6% for betting every under.

**The rule:** bet the **Under** when the model's chance is **0.65 or more**, in passing yards, pass + rush yards, rush attempts, rushing yards, receptions, receiving yards and rush + rec yards. Left out: TD and interception props (books move the price, not the line, on 0.5/1.5 lines, so a win rate at the line says nothing without prices) and pass attempts and completions (negative information against closing lines).

| Frozen rule | Closing lines | Opening lines |
|---|---|---|
| Discovery, ESPN BET 2024-2025 | **56.5%** (n=2,873; about 80 a week) | 58.7% (n=3,124) |
| **Test, DraftKings 2026 weeks 1-6 (untouched)** | **57.4%** (n=284; +27.2 units at -110) | 54.3% (n=727; +27.1 units) |

By market (discovery closing / test closing / test opening): receptions 59.5% / 58.6% / 56.8%; rushing yards 57.3% / 55.2% / 53.3%; rush + rec yards 55.9% / 80% (n=5) / 57.8%; rush attempts 55.2% / 59.1% / 58.7%; receiving yards 53.8% / 40.9% (n=22) / 48.3% (n=172); passing yards 58.8% (n=34) and pass + rush yards 37.0% (n=27) are too small to read. Information against DraftKings closing lines, all markets: 0.085 (t 3.3).

One correction after the first test run: the neighbour window slid inward at the ends of the forecast range (the highest forecasts borrowed only lower ones, which overstated their unders). It now stays centred. Before the fix the same rule scored 56.5% of 2,928 (discovery closing), 56.3% of 293 (test closing) and 54.1% of 736 (test opening). The conclusions are unchanged.

**Hist. Win % and Bet At.** Hist. Win % is the rule's record in the pick's market on closing lines since 2024 (discovery plus every finished week since), shrunk toward the all-market rate by 200 picks' weight (`market_win_rates`), so a market with few picks does not set a price on its own. Bet At is the worst price at which that rate still returns 3% (`MIN_EDGE`). As of week 6 of 2026: -133 for receptions (58.9%), -124 for rushing yards, -121 for rush + rec yards, -119 for rush attempts and -111 for receiving yards (54.1%). Since DraftKings often prices unders at -115 to -130, many receiving-yards picks will not clear their Bet At; that is intended.

**Diagnostics, not used in the rule:** unders on players whose teammates are out (20%+ of the team's targets or carries vacated), where the line may already price a bigger role, won 57.5% of 480 (discovery) and 22 of 29 (test), so they are not a weak spot. Weeks 1-3 are mixed (58.1% in discovery, 50.6% of 168 in the test).

### 11.5 The workbook (`nfl_props_sheet.py`)

`<Desktop>\<year>_nfl_props.xlsx`, rebuilt every run after the Prop Log is read back by header text, saved through a temporary file like the spread workbook:
- **This Week**: the season scoreboard (W-L-P, win %, units at your price or -110, overs W-L, unders W-L, average line versus the close, last week's W-L; all formulas over the Prop Log), then this run's picks best first, in an Over section (bet now, at the opening line) and, when enabled, an Under section (kickoff, game, player, prop, pick, where the line opened, forecast, model chance, Hist. Win %, Bet At, why), the games not evaluated (already kicked off) and a legend.
- **Prop Log**: every pick of the season, logged the first time it qualifies and never rewritten (a re-run adds new picks only, so the log keeps the line available when the pick was made). Type the price you got in **Your Price** and anything else in **Notes** (yellow; both kept across runs). **Units** is a formula on Your Price (blank = -110). Result, W/L (W, L, P or **Void** when a finished game has no nflverse stats for the player: DraftKings voids props for players who do not play) and the **Closing Line** / **Line vs Close** (positive = your line beat the close) fill in once the game is over. Keys and raw numbers are grouped at the right.
- **Model Info**: the rules in words, the live record by market behind each rule's Hist. Win %, and the frozen evidence.

### 11.6 Weak points

1. **No prices.** Every win rate is at the line. Unders are often juiced (-115 to -130), and break-even is 52.4% at -110, 54.5% at -120 and 56.5% at -130. Bet At is the guard, but it relies on the backtest rate per market.
2. **Two books.** Discovery is ESPN BET (a softer book) and the test is DraftKings. ESPN BET's stored lines are taken as pre-game closes on the evidence above. DraftKings' clean closes are a selected subset (lines not updated in-game, mostly receptions and attempts), and the DraftKings openers are early-week numbers the model's late-week features have an advantage over.
3. **Small test.** 284 closing-line picks give a 95% interval of about +/-5.8 points. The next clean evidence is forward: grade the Prop Log and watch Line vs Close (the closing line comes from ESPN when DraftKings did not move it in-game, otherwise from this project's last snapshot before kickoff, so run the props runner close to kickoff to capture it).
4. **Correlated picks.** Many unders come from the same game (a slow game sinks or lifts them together), so weekly results swing more than independent bets would.
5. **Voids versus zeros.** nflverse has no row for a player who played but touched nothing, so such a game is marked Void although DraftKings would grade the under a win.
6. **Models refit once a season.** The passing environment drifts within a season (forecast bias +9 passing yards in 2025); the outcome distributions absorb a steady bias but not a new one.
7. **Receiving yards** was the weakest market in the test. It was kept because the rule was frozen first; its low Hist. Win % already tightens its Bet At.

### 11.7 Overs: the early-week bet (`OVER_*` in `nfl_prop_signal.py`; `python nfl_research_props.py overs` / `overs_test`)

The owner prefers overs, and against closing lines the model's overs carry no information (section 11.4). But against **opening** lines they did (information 0.066, t 5.7, in discovery), which means the market moves toward them during the week. The question was whether that is real edge or look-ahead, since the full model's features include the final injury report and the closing game lines, which nobody has when props open.

**The early-week variant** answers it. It uses only what is known when props open: no injury-report features (blanked in training and forecasting alike) and the **opening** spread and total, which ESPN keeps for every game (`nfl_prop_lines.load_game_openers`, cached in `data_cache/game_openers.csv`; `nfl_prop_model.with_openers`). Seasons before 2024 have no openers and keep the closing numbers; only the models' training uses them. Its own forecast cache is `prop_forecasts_early_v{MODEL_VERSION}.csv`. Pre-registered bar, on the discovery lines (ESPN BET 2024-2025): win 55%+ at the opening line **and** closing-line value with t >= 2 (the line moving toward the over after the bet).

**Discovery findings (ESPN BET 2024-2025 opening lines):**
- The early variant's overs did as well as the full model's (0.65+: 59.5% of 1,245 vs 55.4% of 648), so the edge is not late news.
- Win rate by chance, line-moving markets: 53.5% (0.55+), 55.8% (0.60+), 58.9% (0.65+, n=1,174), 60.5% (0.70+), 68.1% (0.75+).
- **No market drift:** across all lines, 29% rose and 26% fell from open to close. Where the variant leans over at 0.65+, the line rose for 53% and fell for 11%; where it leans under, it fell for 46% and rose for 14%. It predicts the week's line moves in every market.
- **The edge lives in the lines that move:** overs whose line later rose won 67.0% at the opener (54% of the picks). Those that never moved won 52.1% (35%), and those that fell won 40.9% (11%). At the closing line the same overs won about 51-52%.
- For bettors at the close there is nothing for overs: the full model's overs won about 51% whether the line rose or never moved, and 49.8% when it fell.

**The over rule (frozen before the test):** bet the **Over** at DraftKings' opening line when the early-week variant's chance is **0.65 or more** (0.65 matches the under rule; the grid's best-looking 0.75 was not chosen), in every market but passing TDs and interceptions (books move their price, not the line). It applies only **while the line is still at the opening number**.

| Frozen over rule, at opening lines | Win % | Picks |
|---|---|---|
| Discovery, ESPN BET 2024-2025 | **58.9%** | 1,174 (about 33 a week; +145 units at -110) |
| **Test, DraftKings 2026 weeks 1-6 (untouched)** | **55.7%** | 183 (about 30 a week; +11.7 units at -110) |

Test by market: receptions 63.5% (63), receiving yards 55.9% (34), passing yards 54.2% (24); the rest have 15 picks or fewer. Where DraftKings' close is known (73 of the picks), the line rose for 26% and fell for 11% (average move t = 2.4), and at the close those bets won 49.3%. Information of the variant's over leans at DraftKings openers: 0.057 (t 2.1), as in discovery (0.054).

**Late overs.** A line still at its opener close to kickoff is mostly one the market never moves, and those won only 53.2% (457 picks with a known close, discovery and test). An over found within `OVER_LATE_HOURS` (48) of kickoff is therefore marked LATE and priced at that rate, so its Bet At asks for about even money (-107). The 48 hours is a judgment call: the public history has only opening and closing lines, so when lines move during the week is unknown. `prop_line_log.csv` (a snapshot every run) will show it.

**Over-specific weak points:**
1. It only pays if you bet early, before the line moves. Opening prop limits are low (fine at recreational stakes), and the opener may be gone within hours of posting.
2. The test is small: 183 picks, about +/-7 points at 95%. Expect roughly 55-58% at the opener.
3. ESPN BET's openers (discovery) may have been softer than DraftKings' (test), as the drop from 58.9% to 55.7% suggests.
4. Low lines such as Over 0.5 or 1.5 receptions are often heavily juiced. Bet At is the guard (e.g. receptions overs at about -156 early in the week).

## 12. Wind unders (`nfl_wind_unders.py`, October 2026)

A separate game-day check for totals; it shares no logic with the spread runner or the props flow (it only reads the schedule, the stadium coordinates in `nfl_lab_extra.TEAM_SITES`, and the Action Network feed the spread code uses).

**Run it close to kickoff:** Thursday for Thursday games, Saturday night or Sunday morning for Sunday games: `python nfl_wind_unders.py` (the next week with unplayed games) or `python nfl_wind_unders.py 2026 8`. For every home game at an open-air stadium (the schedule's roof is `outdoors`; domes, retractable roofs and neutral sites are skipped) it fetches the Open-Meteo forecast (free, no key), averages the wind at 10 m over the kickoff hour and the three after (the measure the research tested; checked to match the research's forecast archive exactly), shows DraftKings' current total and prices, and flags:
- **UNDER** at 15+ mph: bet the under at the number shown.
- **WATCH** at 10-15 mph: 60% on forecasts since 2018 (169 games), but spotted in that test rather than fixed in advance, so it is tracked separately before it is trusted.
- No flag in weeks 1-4: the effect did not hold there (48% on recorded wind), while weeks 17-18 (65%) and the playoffs (71%) did.

**The season log**, `<Desktop>/<season>_wind_unders.csv`: one row per outdoor game with its latest check before kickoff (forecast wind, total, prices, book, flag) and, on later runs, the final total and the under's result. A game first run after kickoff is not added, so the live record only counts real pre-game checks. The run prints the season record for UNDER and WATCH flags.

**Evidence** (`nfl_research_totals.py`, section 5 item 7): 57% unders at 15+ mph of recorded wind in every era since 1999 (499 games, weeks 5-16, about +10% ROI at the real closing odds); 56% on archived forecasts since 2018 (28-22). The market lowers totals for wind by about half of what scoring falls.

**Weak points:** 50 forecast-based bets is a small sample; the recorded-wind evidence is long but is not what a bettor sees; forecasts two or more days out are rough (rerun near kickoff); the edge was measured at the closing total, which already includes the forecast, so betting near kickoff is enough, and betting earlier risks the forecast changing. Flags at 15+ mph average about 6 a season on the forecast's scale (forecast wind runs about 1.2 mph below recorded stadium wind).

## 13. One-click weekly run (`run_all.bat`, `nfl_run_all.py`, October 2026)

Double-click `run_all.bat` (or the "NFL weekly run" shortcut on the Desktop, which points to it) to run the three weekly flows one after another: `nfl_prod_projections.py` (spread projections, picks, favorite picks, teaser legs), `nfl_prod_props.py` (player props) and `nfl_wind_unders.py` (wind unders). From a terminal in `nfl/`: `python nfl_run_all.py`, or name the flows to run only some, e.g. `python nfl_run_all.py projections wind`.

- **No shared logic.** Each flow runs as its own Python process, exactly as if started by hand: the wrapper imports none of them, passes them no settings (each keeps its own Configuration block, logging and output files), and a failure in one does not stop the others. They run in sequence, not side by side, because the projections and props flows refresh the same nflverse files in `data_cache`.
- **Before starting** it checks that none of the Desktop files the chosen flows write (`<year>_nfl_projections.xlsx`, `<year>_nfl_props.xlsx`, `<year>_wind_unders.csv`) is open in Excel, and asks you to close any that are.
- **While running** it shows everything the flows print as it happens; the full output is also saved to `~/nfl_run_all_<timestamp>.txt` (each flow still writes its own log as before).
- **At the end** a summary lists each flow's result (OK or FAILED with its exit code), time, warning count, the files it updated and any errors worth a look; the window stays open until a key is pressed. The wrapper exits non-zero if any flow failed.
- **Timing:** a full run takes several minutes (the projections' 10,000 bootstrap fits alone take about 2). The props overs want an early-week run and the wind check wants a game-day run, so one click early in the week and a `wind`-only run near kickoff (`run_all.bat wind`) covers both.
