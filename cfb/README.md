# College football research: current state and handoff

## Daily routine (start here)

**Once a day, around 10 AM Eastern, double-click "CFB daily run" on the Desktop** (separate from "NFL daily run"). It makes the spread picks (about 2 minutes), then saves the day's card of new picks in **CFB\Cards** on the Desktop (`2026 week 11 - Sat Nov 7.png`, and the same picks as text in the `.txt` beside it) and rebuilds the card record. **Share the card if the run made one**; when nothing is new it says "No new picks ... nothing to share today" and makes no card. Close the workbooks in Excel first (the run asks if one is open).

- **Everything lives in the CFB folder on the Desktop:** `2026_cfb_picks.xlsx` (This Week and the Season Log, section 4), `2026_cfb_card_record.xlsx` (every card pick, graded at the card's line, with **Bet?**: set it to No for a pick you did not bet), `Cards`, `data` (`share_card_log.csv`: what each card showed) and `logs`.
- **Picks go out at the line on the card and stay there.** A pick follows each run until a card shows it; from the next day on (or at kickoff) the picks workbook keeps it as it was shared, so it is graded at that line. Run it as often as you like: a later run on the same day replaces that day's card.
- **Held back:** a pick whose team's latest game is not in ESPN's play-by-play yet (it comes a day or two after the game) waits for a later card; This Week marks it "Ratings current: No". Saturday's games are usually in by Monday.
- **Another computer:** run `setup.bat` in this folder once (it installs the packages, records `CFB_CODE` and `CFB_PYTHON`, and puts "CFB daily run" on the Desktop), and copy `cfbd_key.txt` (the CollegeFootballData API key, gitignored) into the code folder.
- **Options** (from a terminal in `cfb/`): `python cfb_run_all.py card` remakes the card from the workbook the picks last wrote; `picks` or `nocard` run the picks alone. Each step's output goes to `CFB\logs`.

A separate project from the NFL and NBA suites (`../nfl`, `../nba`): its own code (nothing imported from them) and data cache (`cfb/data_cache`, gitignored). Built November 7, 2026, with the same standards: a rule is fixed on discovery seasons, then checked once on seasons no research step has seen, and judged by what it would have returned at real prices. The older scripts in `../college` (Sports-Reference scrapers) are untouched and not used.

## Where it stands (November 7, 2026)

**The spread model has the strongest signal in the project's three sports, but it did not pass its pre-registered test.** On November 7, 2026 the owner chose to treat it as the NFL picks are treated (bet at an honest expectation of about 55% and judge it on its record), so `cfb_prod_picks.py` now makes weekly picks (section 4). The first look at 2026 is poor: 21-26 through week 10 (below).

- **Spreads** (section 3): the correction model's information about the cover margin is real and steady: r = +0.068 (t 5.6) on the 2010-2019 discovery seasons, positive in all ten, and +0.059 (t 3.5) on the untouched 2021-2025 test (the NBA's was +0.049, the NFL's about +0.07). The rule frozen on discovery, **bet the side the correction favors by 2+ points**, went 55.3% on discovery (433-350) and **56.0% on the test (191-150), +6.9% a bet at the books' median price**, about 68 picks a season, above break-even in 4 of the 5 test seasons. It fails the bar set in advance only on volume: 341 test picks give z = +1.34 against -110 break-even, and the bar was z >= 2. Pooled with discovery it is 624-500 (55.5%, z about 2.1), but the bar was set on the test alone.
- **Totals:** the total correction is weak (r = +0.025, t 2.1 on discovery, negative in three seasons) and no totals rule with 60+ picks a season beat break-even (best 51.9%), so none was chosen and the totals' test seasons are unspent.
- **Model A** (stats only) misses the margin by 13.09 points against the closing spread's 12.33, so the books are better than the ratings alone, as in the NFL and NBA.

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

**Checks** (the full table, 16,851 FBS games 2004-2026, built in about a minute): 12,488 regular-season games 2006-2025 have a closing spread and a result (about 700-760 a season from 2011; 2-3 books a game and no Pinnacle from 2020); home teams covered 49.4% and overs hit 49.0%; the closing spread misses the margin by 10.9-13.0 points a season. On 2010-2019 the pre-game margin rating correlates +0.65 with the final margin (the closing spread +0.69), points per drive +0.52, EPA per play +0.43. No feature is blank. The 2026 season (502 games played by November 7) has results but no lines (section 5).

## 3. Spreads and totals (`cfb_model.py`, `cfb_research_validate.py`)

- **Model A** (stats only, ridge): the margin and total from the ratings; a reference, as in the other suites.
- **Model B** (the correction model, ridge on standardized features): predicts the cover margin and the over margin at the closing line, each season fitted only on earlier seasons from 2006.
- **The pre-registered validation** (written before any result was seen; the protocol is the docstring of `cfb_research_validate.py`): discovery 2010-2019, untouched test 2021-2025, 2020 (the COVID season) used only to train the test's models; regular season only (bowls are left out, since players sit out for the draft and transfers, which the data does not record). A rule passes only if on the test its picks beat -110 break-even by z >= 2, its correction correlates with the outcome with t >= 2, and it beats break-even in at least 4 of the 5 test seasons: stricter than the NBA's bar, which let through rules worth about nothing at -110.

**Discovery** (2010-2019, 6,640 regular-season games with a closing spread; run once):

| Alpha | Spreads r (t) | Totals r (t) |
|---|---|---|
| 300 | +0.059 (4.8) | **+0.025 (2.1)** |
| 1,000 | +0.065 (5.3) | +0.025 (2.1) |
| 3,000 | **+0.068 (5.6)** | +0.025 (2.1) |
| 10,000 | +0.067 (5.4) | +0.023 (1.9) |
| 30,000 | +0.058 (4.7) | +0.018 (1.5) |

Spread rules at alpha 3,000 (win rate by threshold, t0 / t1 / t2 / t3): S1 (either side) 51.6% / 53.2% / **55.3%** / 63.3% (120 picks, too few); S2 (underdogs) 51.2% / 54.7% / 55.3% / 69.0% (42); S3 (favorites) 51.7% / 51.8% / 54.9% / 60.3% (78). The chosen rule, S1 at 2 (433-350), was committed in `CHOSEN` before the test ran (git history: `bfcc2f8`). Totals: no rule with 60+ picks a season beat break-even.

**Test** (2021-2025, untouched until then, each season by models fitted on 2006 to the season before, 2020 included):

| Season | S1 at 2 |
|---|---|
| 2021 | 36-36, 50.0% |
| 2022 | 39-26, 60.0% |
| 2023 | 38-22, 63.3% |
| 2024 | 43-35, 55.1% |
| 2025 | 35-31, 53.0% |
| **All** | **191-150, 56.0%** (z +1.34; +6.9% a bet at the median price) |

Correction versus the cover margin: r = +0.059 (t 3.5). Passes two of the three conditions (t >= 2, 4+ of 5 seasons) and fails the third (z >= 2): **not proposed for betting** by the protocol.

**Forward, 2026** (`python cfb_research_validate.py forward`; first scored November 7, 2026, through week 10, with CollegeFootballData's closing lines, the median of DraftKings and Bovada; models fitted on 2006-2025): the correction kept its information (r = +0.068 on 512 games, t 1.5, about what the history leads to expect), but the rule went **21-26 (44.7%)**: weeks 1-2 0-5, then 21-21. 47 picks say little either way (at a true 55%, a start this bad or worse happens about one season in fifteen); the rule is not changed because of it.

**The NFL comparison.** The NFL underdog rule went into production on a weaker untouched test (53.7-54.0% on 1999-2005, z about 0.6-0.7) because it held at about 55% in every era with steady information; it was never held to z >= 2. By that standard the college rule (56.0%, z 1.34) qualifies at least as well, which is why the owner chose to bet it the same way.

## 4. The weekly picks (`cfb_prod_picks.py`)

`python cfb_prod_picks.py` from this folder (the daily run's first step), about 2 minutes. It refreshes this season's play-by-play and CollegeFootballData's games and lines, fits Model B on every regular season from 2006 to last season (alpha 3,000, as in the test), and picks the side the correction favors by 2+ points in every FBS-vs-FBS regular-season game with a line. Each pick is at DraftKings' line (the books' median when DraftKings has none). It writes `<Desktop>\CFB\<season>_cfb_picks.xlsx`:
- **This Week:** the picks among the games still to kick off (kickoff, game, pick, lean) and the season's record.
- **Season Log:** every pick this season with the line it went out at, the closing line, line versus close (positive: the pick beat the closing line, the quickest sign of a real edge), the final margin and the result. A pick follows the latest run until its game kicks off and is then kept, so it is graded at the line the last run before kickoff showed. Weeks played before the first run were filled in at the closing line and are marked so.

The run log goes to `<Desktop>\CFB\logs`. **CollegeFootballData's API key** is in `cfb/cfbd_key.txt` (gitignored; replace the text to change keys); the free tier allows 1,000 calls a month and a run uses 4.

## 5. Files (`cfb/`)

`cfb_common.py` (paths, season numbering: a season is the year it starts; the API key; the share log), `cfb_data.py` (schedules, groups, the play-by-play reduction), `cfb_cfbd.py` (CollegeFootballData: this season's games, lines and team colors), `cfb_lines.py` (closing lines from the multi-book archive to 2025, CollegeFootballData from 2026), `cfb_lab.py` (the research table and walk-forward harness), `cfb_model.py` (Models A and B), `cfb_research_validate.py` (the pre-registered test and the forward test, with their results in its docstring), `cfb_prod_picks.py` (the weekly picks), `cfb_share_card.py` (the day's card; its drawing follows `nfl_share_card.py`, copied rather than imported so the suites stay separate), `cfb_card_record.py` (the card record), `cfb_run_all.py`, `run_all.bat`, `desktop_launcher.bat` and `setup.bat` (the daily run), `requirements.txt` (the NFL suite's pinned packages). Run them from this folder.

## 6. If this is picked up again

- **Track S1 at 2 forward, unchanged.** The weekly run's Season Log is the live record; `python cfb_research_validate.py forward` rescores the season at closing lines. Volume is the obstacle: at 56% a pick, z = 2 against -110 break-even takes about 760 picks, about six more seasons at 68 a season on top of the 341 test picks. Line value against the close (Season Log) will say sooner whether the edge is real.
- **Totals** have their test seasons unspent; a new idea for them (weather, which the NFL suite found priced at about half its effect; pace mismatches; referee crews) would need its own pre-registered test.
- **Not tried:** recruiting talent and returning production (sportsdataverse's `cfb_team_talent` and `cfb_returning_production` are parquet-only, which this 32-bit Python cannot read), early-season games where ratings carry over most, bowls (left out), FCS opponents, opening lines (DraftKings, Bovada and ESPN Bet from 2023 only), and Pinnacle against the other books (2010-2019).
