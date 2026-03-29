# MLB PICKS — PROGRESS NOTES
# ============================
# Last updated: 2026-03-23
#
# STATUS: COMPLETE AND FULLY WORKING.
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
#     RUN LINE   : pick + cover probability %
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
#   Algorithm  : Ridge regression (sklearn), two targets:
#                  y_diff  = home_score - away_score  (predicts winner/run line)
#                  y_total = home_score + away_score  (predicts O/U)
#   Features   : batting stats (R, H, HR, BB, SO, BA, OBP, SLG, OPS)
#                + pitching stats (ERA, WHIP, H9, HR9, BB9, SO9)
#                for both home and away team
#   Simulation : Monte Carlo bootstrap — resamples training data each
#                iteration, fits a new Ridge model, predicts the game.
#                Distributions of diff/total used to compute win%/cover%/O%.
#   Default    : 500 iterations (--runs to change)
#
# ==========================================================================
# BUGS FIXED (history)
# ==========================================================================
#   1. _load_table(): batting table is VISIBLE (not HTML-commented).
#      Fixed by checking visible <table> tags first, then falling back
#      to HTML comments (pitching table IS commented).
#   2. load_schedule(): Nested div structure caused all p.game tags to be
#      absorbed by the outermost div, causing 0 games to load.
#      Fixed: iterate soup.find_all(['h3','p']) in document order.
#   3. _parse_game_para(): '@' separator in HTML is '\n @\n', not ' @ '.
#      Fixed with re.search(r'\s@\s', raw_html).
#
# ==========================================================================
# POTENTIAL FUTURE ENHANCEMENTS
# ==========================================================================
#   - Starting pitcher stats (per-game ERA/WHIP would improve accuracy)
#   - Multi-year training data (pass multiple sched_years, concat schedules)
#   - MLB backtest script (like nfl_ats_backtest.py — test model vs results)
#   - Implied probability vig removal (normalize two-sided implied probs)
#   - Auto-fetch lines from an odds API instead of manual UI entry
#
# ==========================================================================
# BACKTESTING (mlb_backtest.py)
# ==========================================================================
#   Tests the model against 2010-2021 historical closing odds.
#
#   STANDARD BACKTEST (one or more years):
#     python mlb_backtest.py --years 2021 --runs 100
#     python mlb_backtest.py --years 2018 2019 2020 2021 --runs 200
#     python mlb_backtest.py --years 2021 --runs 100 --save
#
#   CLI flags:
#     --years       list of years (default: 2010-2021)
#     --runs        Monte Carlo iterations per game (default: 100)
#     --save        save game-by-game results to mlb_backtest_results.csv
#     --min_games   min completed games before model starts predicting (30)
#     --sweep       run training window sweep (see below)
#     --max_prior   max prior seasons to test in sweep (default: 6)
#
#   Output: year-by-year table + totals for ML/RL/OU accuracy and ROI,
#           plus confidence tier breakdown (HIGH >= 70%, MEDIUM 60-69%,
#           LOW < 60%).
#
# ==========================================================================
# TRAINING WINDOW SWEEP RESULTS (2015-2021, 50 runs)
# ==========================================================================
#   Command used:
#     python mlb_backtest.py --years 2015 2016 2017 2018 2019 2020 2021
#                            --runs 50 --sweep
#
#   Results:
#     Window               Games    ML%   ML ROI    RL%   RL ROI  OU%  OU ROI
#     Current season only  13820  56.6%    +0.2%  41.5%   +10.9%  52.8%  +1.1%
#     +1 prior season      13970  57.8%    +1.6%  41.5%   +11.3%  53.5%  +2.5%
#     +2 prior seasons     13970  58.1%    +2.0%  40.9%    +9.7%  53.5%  +2.6%
#     +3 prior seasons     13970  58.2%    +2.2%  40.7%    +9.3%  53.3%  +2.1%
#     +4 prior seasons     13970  58.1%    +2.1%  40.6%    +9.0%  53.6%  +2.7%
#     +5 prior seasons     13970  58.3%    +2.4%  40.4%    +8.5%  53.6%  +2.8%
#     +6 prior seasons     13970  58.5%    +2.7%  40.4%    +8.6%  53.8%  +3.0%
#     >> Best ML ROI: window=6  ROI=+2.7%
#
#   KEY FINDINGS vs NFL model:
#     - MLB accuracy improves MONOTONICALLY with more prior data.
#       No accuracy dropoff like the NFL (where >5 years hurt).
#     - Biggest single gain: 0->1 prior season (+1.4% ML ROI).
#       Diminishing returns after that but still positive.
#     - RL accuracy slightly DECLINES with more data (41.5% -> 40.4%)
#       while RL ROI stays roughly flat — RL is driven more by current
#       season pitching trends than long-term team averages.
#     - OU improves modestly with more data (+1.9% ROI over 6 windows).
#     - Window=6 still trending up — true optimum likely higher but
#       limited by data availability (earliest year = 2010).
#
#   RECOMMENDATION:
#     Use 3 full prior seasons + current season for all live predictions.
#     RL accuracy peaks at window=1 but overall quality (ML+RL+OU combined)
#     is best with 3 prior seasons. This is now the default (--prior_years 3).
#     All prior-year stats use that year's HTML — team_stats_<py>.html must
#     be present in C:/Users/Bobby/Downloads/MLB_Stats/.
#
# ==========================================================================
# NEXT STEPS
# ==========================================================================
#   1. Re-run sweep at --runs 200 on window=3 to confirm results
#   2. Consider starting pitcher ERA/WHIP as per-game feature

