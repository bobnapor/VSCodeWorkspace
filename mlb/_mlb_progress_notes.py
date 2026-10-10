# MLB PICKS — PROGRESS NOTES
# ============================
# Last updated: 2026-03-29
#
# STATUS: WORKING — run-line display issue under investigation
#         (see NEXT STEPS section at bottom).
#   mlbpicks.py      — core prediction platform
#   mlb_lines_ui.py  — tkinter UI for daily lines entry
#   mlb_backtest.py  — backtesting framework + training window sweep
#
# ==========================================================================
# FILE INVENTORY
# ==========================================================================
#   mlbpicks.py              -- main prediction script
#   mlb_lines_ui.py          -- daily matchup/odds entry UI (tkinter)
#   _mlb_progress_notes.py   -- this file
#   mlb_backtest.py          -- backtesting vs historical odds 2010-2021
#
# ==========================================================================
# DATA FILES  (C:/Users/Bobby/Downloads/MLB_Stats/)
# ==========================================================================
#   team_stats_<yyyy>.html
#     batting + pitching stats from baseball-reference.com
#   schedule_<yyyy>.html
#     full season schedule from baseball-reference.com
#   NOTE: 2010-2014 files use older prefix: stats_<yyyy>.html
#         (mlbpicks.py auto-fallback handles this)
#
#   Historical odds (backtest only):
#     mlb-odds-<yyyy>.xlsx   (2010-2021, from sportsbookreviewsonline.com)
#     Columns: Date, Rot, VH, Team, Pitcher, 1st-9th innings, Final,
#              Open, Close (ML), RunLine, RL odds, OpenOU, OU odds, CloseOU
#     Note: 2010 file has no RunLine column; different O/U column names.
#
# ==========================================================================
# HOW TO USE — STANDARD (schedule-based)
# ==========================================================================
#   python mlbpicks.py --date 2025-08-20
#   python mlbpicks.py --date 2025-08-20 --runs 500
#   python mlbpicks.py --date 2025-08-20 --save       # saves CSV to C:/Users/Bobby/
#   python mlbpicks.py                                 # prompts for date
#
#   Output includes per-game:
#     MONEYLINE  : pick + model confidence %  + tier (HIGH/MEDIUM/LOW)
#     RUN LINE   : most likely of 3 outcomes + its probability %
#                    "TeamName -1.5"              team covers by >1.5 runs
#                    "Push zone (<=1 run margin)" margin stays within 1 run
#     OVER/UNDER : (only shown when --lines CSV is provided)
#     PROJ SCORE : projected score for each team
#
# ==========================================================================
# HOW TO USE — WITH DAILY LINES (recommended for real betting days)
# ==========================================================================
#   Step 1:  python mlb_lines_ui.py
#            Enter today's matchups, run lines, ML odds, O/U lines.
#            Click "Save CSV" -> writes to C:/Users/Bobby/mlb_lines.csv
#
#   Step 2:  python mlbpicks.py --date 2026-03-22 --lines C:/Users/Bobby/mlb_lines.csv
#
#   With --lines, the output also shows:
#     - Book ML odds vs model win% as an EDGE indicator
#       e.g.  Book ML: Mets -165 (implied 62.3%)  edge +29.2%
#             positive edge = model sees more value than the book prices
#     - O/U picks with confidence % against the real line
#     - Per-game run lines (if line differs from standard 1.5)
#
# ==========================================================================
# CLI ARGUMENTS
# ==========================================================================
#   --date        YYYY-MM-DD  prediction date (default: prompt)
#   --runs        int         Monte Carlo iterations (default: 500)
#   --stats_year  int         year for team_stats HTML (default: 2025)
#   --sched_year  int         year for schedule HTML (default: 2025)
#   --lines       path        path to mlb_lines.csv from mlb_lines_ui.py
#   --prior_years int         full prior seasons to add to training data
#                             (default: 3 — uses 3 full prior seasons +
#                              current season up to prediction date)
#                             set to 0 for current season only
#   --save                    save picks to CSV at C:/Users/Bobby/mlb_picks_<date>.csv
#
# ==========================================================================
# ADDING MORE YEARS
# ==========================================================================
#   Save HTML files and name them:
#     team_stats_2026.html
#     schedule_2026.html
#   Then run:
#     python mlbpicks.py --date 2026-07-04 --stats_year 2026 --sched_year 2026
#
# ==========================================================================
# MODEL OVERVIEW
# ==========================================================================
#   Algorithm  : Ridge regression, two targets:
#                  y_diff  = home_score - away_score  (winner / run line)
#                  y_total = home_score + away_score  (over/under)
#   Features   : batting stats (R, H, HR, BB, SO, BA, OBP, SLG, OPS)
#                + pitching stats (ERA, WHIP, H9, HR9, BB9, SO9)
#                for both home and away team (32 features total)
#                + starting pitcher ERA/WHIP/K9/BB9 (blended current/prior)
#   Simulation : Bootstrap Monte Carlo — vectorized via numpy einsum +
#                normal equations. Dynamic chunk sizing <60MB/allocation.
#                Distributions of diff/total used for win%/cover%/O%.
#   Default    : 500 iterations (--runs; use 5000 for production)
#
# ==========================================================================
# PITCHER STAT BLENDING
# ==========================================================================
#   _lookup_sp(name, pitcher_stats, prev_pitcher_stats) blends by GS:
#     GS >= 5 (current season)  -> 100% current season stats
#     GS 1-4 (current season)   -> weighted blend: w_cur = GS/5
#     GS = 0 / not in current   -> 100% prior season (if found)
#     Not found in either year  -> average of both years' _DEFAULT medians
#   Starters fetched from MLB Stats API if not in schedule HTML.
#   Blending threshold constant: MIN_BLEND_GS = 5
#
# ==========================================================================
# PERFORMANCE — VECTORIZED BOOTSTRAP ENGINE
# ==========================================================================
#   _bootstrap_predict(X_np, y, X_pred_np, num_runs, alpha=1.0)
#     - Draws all (num_runs, n) bootstrap indices at once
#     - Dynamic CHUNK = max(1, min(50, 60MB // (n*p*8 bytes)))
#     - Batched: XtX = einsum('bni,bnj->bij', X_aug, X_aug)
#                coeffs = linalg.solve(XtX, Xty)
#     - Returns (g, num_runs) float64 array — no Python loop per iter
#     - 2026-03-29 run: n=13307, p=32, chunk=18 (~58MB/alloc)
#     - Verified identical to sklearn Ridge (max coeff diff: 5.55e-17)
#
# ==========================================================================
# BUGS FIXED (history)
# ==========================================================================
#   1. _load_table(): batting table is VISIBLE (not HTML-commented).
#      Fixed by checking visible <table> tags first.
#   2. load_schedule(): nested div structure caused 0 games to load.
#      Fixed: iterate soup.find_all(['h3','p']) in document order.
#   3. _parse_game_para(): '@' separator is '\n @\n', not ' @ '.
#      Fixed with re.search(r'\s@\s', raw_html).
#   4. _parse_date_header(): "Today's Games" header caused crash.
#      Fixed: if 'today' in text.lower(): return date.today()
#   5. Run-line displayed wrong side (underdog +1.5 instead of fav -1.5).
#      Fixed: compare home_cover_pct vs away_cover_pct, show -1.5 side.
#   6. Run-line now shows MOST LIKELY of 3 outcomes:
#        "TeamName -1.5", "TeamName -1.5" (away), "Push zone (<=1 run)"
#        max(home_cover_pct, away_cover_pct, push_pct) picks the label.
#   7. UnicodeEncodeError on Windows cp1252 for "≤" char.
#      Fixed: use ASCII "<=" in push zone label string.
#
# ==========================================================================
# BACKTESTING (mlb_backtest.py)
# ==========================================================================
#   Tests the model against 2010-2021 historical closing odds.
#
#   STANDARD BACKTEST:
#     python mlb_backtest.py --years 2021 --runs 100
#     python mlb_backtest.py --years 2018 2019 2020 2021 --runs 200
#     python mlb_backtest.py --years 2021 --runs 100 --save
#
#   CLI flags:
#     --years       list of years (default: 2010-2021)
#     --runs        Monte Carlo iterations per game (default: 100)
#     --save        save results to mlb_backtest_results.csv
#     --min_games   min completed games before predicting (default: 30)
#     --sweep       run training window sweep
#     --max_prior   max prior seasons in sweep (default: 6)
#
# ==========================================================================
# TRAINING WINDOW SWEEP RESULTS (2015-2021, 50 runs)
# ==========================================================================
#   Command:
#     python mlb_backtest.py --years 2015 2016 2017 2018 2019 2020 2021
#                            --runs 50 --sweep
#
#   Results:
#     Window               Games    ML%  ML ROI   RL%  RL ROI  OU%  OU ROI
#     Current season only  13820  56.6%   +0.2%  41.5%  +10.9%  52.8%  +1.1%
#     +1 prior season      13970  57.8%   +1.6%  41.5%  +11.3%  53.5%  +2.5%
#     +2 prior seasons     13970  58.1%   +2.0%  40.9%   +9.7%  53.5%  +2.6%
#     +3 prior seasons     13970  58.2%   +2.2%  40.7%   +9.3%  53.3%  +2.1%
#     +4 prior seasons     13970  58.1%   +2.1%  40.6%   +9.0%  53.6%  +2.7%
#     +5 prior seasons     13970  58.3%   +2.4%  40.4%   +8.5%  53.6%  +2.8%
#     +6 prior seasons     13970  58.5%   +2.7%  40.4%   +8.6%  53.8%  +3.0%
#     >> Best ML ROI: window=6  ROI=+2.7%
#
#   KEY FINDINGS:
#     - ML accuracy improves monotonically with more prior data.
#     - RL accuracy slightly DECLINES with more data (41.5% -> 40.4%)
#       while RL ROI stays roughly flat.
#     - OU improves modestly (+1.9% ROI over 6 windows).
#     - Window=6 still trending up; true optimum likely higher.
#
#   RECOMMENDATION:
#     Use --prior_years 5 for live predictions (best overall balance).
#     Code default is 3; production runs use 5.
#
# ==========================================================================
# 2026-03-29 PRODUCTION RUN — OBSERVATIONS
# ==========================================================================
#   Command: python mlbpicks.py --runs 5000 --date 2026-03-29 --prior_years 5
#   Training: 13,307 games (2477 current + 5 prior seasons)
#   Bootstrap: n=13307, p=32, chunk=18 (~58MB/alloc)
#   12 Opening Day games predicted successfully.
#
#   ISSUE: Run-line shows "Push zone (<=1 run margin) 100.0%" for 9/12 games.
#
#   ROOT CAUSE (diagnosed, not yet fixed):
#     Ridge regression shrinks predicted run differentials toward zero.
#     The bootstrap diffs distribution has very low variance — most
#     samples fall in (-1.5, +1.5), so push_pct is genuinely ~100%.
#     This was always true but was HIDDEN by the old display logic which
#     showed the -1.5 side even when cover% was 0.1%.
#     Now that we show the TRUE most-likely outcome, the compressed
#     predictions are exposed.
#     Moneyline is unaffected (only needs diffs > 0).
#
# ==========================================================================
# NEXT STEPS — IMMEDIATE PRIORITY
# ==========================================================================
#
#   1. DIAGNOSE run margin compression  <-- START HERE
#      Add a one-time debug print of diffs distribution for one game:
#        mean, std, 10th/25th/50th/75th/90th percentiles
#      Expected for a real MLB game: std ~2.5-3.5 runs.
#      If std <<1.0, ridge alpha is over-shrinking the coefficients.
#
#   2. TUNE ridge alpha (regularization strength)
#      Current: alpha=1.0 (default in _bootstrap_predict).
#      Try: alpha=0.01 or alpha=0.1 (less shrinkage = larger margins).
#      _bootstrap_predict() already accepts the alpha param — just
#      need to wire it through predict_games() -> main() -> CLI arg.
#      Suggested: add --alpha float CLI argument.
#
#   3. VERIFY with backtest after tuning
#      python mlb_backtest.py --years 2021 --runs 200
#      Confirm RL accuracy improves, ML doesn't degrade.
#
#   4. CONSIDER feature scaling (optional)
#      StandardScaler on X before fitting may help ridge produce
#      better-calibrated coefficients. Currently no scaling applied.
#
# ==========================================================================
# POTENTIAL FUTURE ENHANCEMENTS
# ==========================================================================
#   - Auto-fetch lines from an odds API instead of manual UI entry
#   - Implied probability vig removal (normalize two-sided implied probs)
#   - Re-run sweep at --runs 200 to tighten window recommendation
#   - Bullpen/relief stats as additional features

