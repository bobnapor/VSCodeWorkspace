# College football research: current state and handoff

A separate project from the NFL and NBA suites (`../nfl`, `../nba`): its own code (nothing imported from them) and data cache (`cfb/data_cache`, gitignored). Built November 7, 2026, with the same standards: a rule is fixed on discovery seasons, then checked once on seasons no research step has seen, and judged by what it would have returned at real prices. The older scripts in `../college` (Sports-Reference scrapers) are untouched and not used.

## Where it stands (November 7, 2026)

RESULTS_PLACEHOLDER

## 1. Data (all free, no API key; `cfb_data.py`, `cfb_lines.py`)

| Source | Content |
|---|---|
| sportsdataverse's `espn_cfb_pbp` (`play_by_play_<season>.csv`) | Every play since 2004 with ESPN's expected points (EPA), success, explosive plays, havoc, line yards, stuffs, win probability, drives and their results, and ESPN's own spread and total for the game. About 550 MB a season with 507 columns; **reduced to one row per team per game** (`data_cache/pbp_team_<season>_v2.csv`, sums and counts) plus one per game (`pbp_games_<season>_v2.csv`), and the raw file deleted, as the NFL suite does with nflverse. This 32-bit Python cannot parse a season at once (the tokenizer runs out of memory at 50,000 rows of 507 columns), so it is read 10,000 plays at a time: about 7 seconds a season after a 30-50 second download. `python cfb_data.py [first [last]]` builds the cache; a full build is about 20 minutes. |
| sportsdataverse's `espn_cfb_schedules` | Every game: week, season type (2 regular, 3 postseason), neutral site, conference game, both teams (ESPN ids), score, status. |
| sportsdataverse's `cfb_groups` (`cfb_team_group_seasons_<season>.csv`) | Each team's subdivision (`cfb:fbs`, `cfb:fcs`, ...) and conference by season, keyed by ESPN team id. Decides which games are FBS against FBS. |
| cfbfastR's `cfb_line_odds.csv.gz` (CollegeFootballData's archive) | 1.18 million rows, 2006-2025: every book's spread, total and moneyline with prices, one row per side per book, keyed by ESPN game id. Many offshore books through 2019 (5Dimes, Bookmaker, Pinnacle, bet365, Bovada, ...), then few (DraftKings, ESPN Bet, Bovada, William Hill); opening lines only for DraftKings, Bovada and ESPN Bet from 2023. |

**Orienting the books' spreads.** The archive names teams by the book's abbreviation (`UTH`, `BAMA`, `NC`), which does not match ESPN's, and a book's two spread rows come in no fixed order (the second row was the home team's 48% of the time). Each season, every abbreviation is matched to an ESPN team by voting: in games where ESPN's own spread is 3+ points, the row with ESPN's sign is the home team's (98% of abbreviations point to one team at least 95% of the time; those under 80% are dropped). In 2019, close games (ESPN's spread under 3) oriented this way put the home line within 2 points of ESPN's in 95.6% of games. `closing_lines` then takes the median across sportsbooks (leaving out consensus, teamrankings and numberfire), keeps Pinnacle's line on its own, and the median opening line where reported.

**Checks** (2006-2011, FBS and FCS games with lines): the books' median closing spread misses the final margin by 11.86 points (Pinnacle 11.87, ESPN's own 12.29); home teams covered 49.5% and overs hit 48.8%. ESPN's early lines include placeholders (a flat -2.5 on games the books had at 29.5), which the voting overrides. The play-by-play reduction against ESPN's team box scores (2019, all 1,780 team-games): yards correlate 0.97, first downs 0.95 (the box score also counts first downs by penalty), turnovers 0.96, rushes 0.98 (sacks count as passes here), drive points with the final score 0.97.

**Drive results** are labelled differently by era: `MADE FG` (2004), `RUSHING TD` / `PASSING TD` / `FG GOOD` (to 2013), `TD` / `FG` (since); some drives are `Not provided`. The first build matched only the newest labels and left points per drive at 0 before 2014 (caught by its correlation with the margin, +0.01); `PBP_VERSION` 2 matches all of them.

## 2. The research table (`cfb_lab.py`)

One row per game between two FBS teams (FCS games still feed the ratings), with everything known before kickoff, home minus away (`d_`) and home plus away (`s_`); `default_table()` caches it (`data_cache/cfb_table_v1.pkl`).
- **Team-game stats** (`team_stats`, 25, offense and defense): points per drive, EPA per play (all, passes, runs, competitive plays with the offense's win probability 10-90%, early downs), success rates (all, passes, runs, third and fourth down), explosive plays, yards per play, line yards and stuffs per run, sacks per pass, havoc per play, turnovers and scoring opportunities (drives reaching the 40) per drive, points per opportunity, starting field position, special-teams EPA, penalty yards, points, plays and margin.
- **Ratings** (`rate_teams`), as in the NBA suite: a recency-weighted average of earlier games (0.90 a game), each corrected by the opponent's deviation from the league average, shrunk toward the tier's mean by 3 pseudo-games, carried into a new season at 50%. College schedules differ far more than the pros' (a top-10 opponent one week, an FCS team the next), so the opponent correction is full (1.0, against 0.6 in the NFL and NBA), and non-FBS teams are shrunk toward the non-FBS mean. Built in date order from earlier games only; same-day games do not see each other.
- **Context:** rest days (capped at 21), games played this season, home field (not at neutral sites), conference game.
- **Lines and results:** the closing lines (section 1), the margin and total, the cover margin (`margin + home spread`) and the over margin (`points - total`).

## 3. Spreads and totals (`cfb_model.py`, `cfb_research_validate.py`)

- **Model A** (stats only, ridge): the margin and total from the ratings; a reference, as in the other suites.
- **Model B** (the correction model, ridge on standardized features): predicts the cover margin and the over margin at the closing line, each season fitted only on earlier seasons from 2006.
- **The pre-registered validation** (written before any result was seen; the protocol is the docstring of `cfb_research_validate.py`): discovery 2010-2019, untouched test 2021-2025, 2020 (the COVID season) used only to train the test's models; regular season only (bowls are left out, since players sit out for the draft and transfers, which the data does not record). A rule passes only if on the test its picks beat -110 break-even by z >= 2, its correction correlates with the outcome with t >= 2, and it beats break-even in at least 4 of the 5 test seasons: stricter than the NBA's bar, which let through rules worth about nothing at -110.

TEST_PLACEHOLDER

## 4. Files (`cfb/`)

`cfb_common.py` (paths, season numbering: a season is the year it starts), `cfb_data.py` (schedules, groups, the play-by-play reduction), `cfb_lines.py` (closing lines from the multi-book archive), `cfb_lab.py` (the research table and walk-forward harness), `cfb_model.py` (Models A and B), `cfb_research_validate.py` (the pre-registered test, with its results in its docstring). Run them from this folder with the NFL suite's packages (`../nfl/requirements.txt`).

## 5. If this is picked up again

NEXT_PLACEHOLDER
