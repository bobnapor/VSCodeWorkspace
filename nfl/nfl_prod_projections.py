"""Weekly NFL projections.

Projects every game of one week with the nflverse-based model in nfl_model.py, logs the projections, and
adds them to the season spreadsheet (nfl_sheet.py). Run it from this folder:  python nfl_prod_projections.py
"""

import logging
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np

import nfl_lab as lab
import nfl_model as nm
import nfl_sheet as sheet
from nfl_injury_features import TEAM_FULL_NAMES

# Team stats the model rates (offense and defense each): net yards, turnovers, first downs, points and plays
MODEL_METRICS = nm.PRODUCTION_METRICS
# How many times the projection regression is refitted, each time on a bootstrap resample of the training games (drawn
# with replacement); the projection is the average of the refits and StdDev their spread.  More runs make that average
# and spread steadier (they settle within a few hundred); they do not change the model or the picks.  About 70 seconds
# for 10,000.
BOOTSTRAP_RUNS = 10000
TOTAL_STEPS = 8

PROJECTION_HEADER = "Game|Week|Year|Home|Away|Mean|FairLine(Home)|StdDev|Actual"


class RunLog:
    """Numbered, timed steps in the log, so a long run can be followed as it goes."""

    def __init__(self, total_steps=TOTAL_STEPS):
        self.total, self.started = total_steps, time.time()

    def elapsed(self):
        return time.time() - self.started

    @contextmanager
    def step(self, number, name):
        label = f"[{number}/{self.total}] {name}"
        logging.info(f"{label}...")
        began = time.time()
        yield
        logging.info(f"{label}: done in {time.time() - began:.1f}s ({self.elapsed():.0f}s since start)")


def week_label(game_type, week):
    """Week as shown in the sheet: a number for the regular season, a round name for the playoffs."""
    return sheet.PLAYOFF_WEEK_NAMES.get(game_type, str(int(week)))


def find_next_week_to_predict(table, season):
    """Earliest regular-season week of `season` that still has unplayed games, else the last week played."""
    regular = table[(table['season'] == season) & (table['game_type'] == 'REG')]
    unplayed = regular[regular['result'].isna()]
    if not unplayed.empty:
        return int(unplayed['week'].min())
    played = regular[regular['result'].notna()]
    if played.empty:
        raise ValueError(f"No games found for {season}; cannot determine the week to predict")
    logging.warning(f"No unplayed games left in {season}; using last played week {int(played['week'].max())}")
    return int(played['week'].max())


def project_week(table, season, week, runs=BOOTSTRAP_RUNS, progress=None):
    """Projected home margin for each game of a week: mean, median and spread over `runs` bootstrap fits.

    `progress(done, runs)` is called about every 10% of the fits.
    """
    games = table[(table['season'] == season) & (table['week'] == week)].sort_values('gameday').reset_index(drop=True)
    if games.empty:
        raise ValueError(f"No games found for {season} week {week}")
    train = nm.training_frame(table, season, before_week=week)
    columns = nm.feature_columns(MODEL_METRICS, use_qb=True, use_elo=True)
    draws = nm.bootstrap_projection(train, games, columns, runs, progress=progress)
    games['mean'] = draws.mean(axis=0)
    games['median'] = np.median(draws, axis=0)  # computed but not printed
    games['std'] = draws.std(axis=0)
    return games, len(train)


def main(num_runs=BOOTSTRAP_RUNS, predict_year=None, predict_week=None):
    """Project one week. predict_year / predict_week default to the current season and its next unplayed week."""
    log_filename = f'C:/Users/Bobby/nfl_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
        force=True,
    )
    logging.info(f"Logging to file: {log_filename}")
    run = RunLog()

    season = predict_year or sheet.get_current_season_year()
    logging.info(f"Run settings: {num_runs:,} bootstrap regressions for the projections; progress is logged in {TOTAL_STEPS} steps")
    with run.step(1, "Loading nflverse data (schedules, team stats, injuries)"):
        table = nm.load_game_table(refresh=True)
    if not (table['season'] == season).any():
        raise ValueError(f"nflverse has no {season} schedule yet")
    week = predict_week or find_next_week_to_predict(table, season)
    logging.info(f"Predicting: Year {season}, Week {week}")

    with run.step(2, f"Fitting {num_runs:,} bootstrap regressions for the projections"):
        began = time.time()

        def progress(done, total):
            elapsed = time.time() - began
            logging.info(f"    {done:,} of {total:,} regressions ({done / total:.0%}), {elapsed:.0f}s elapsed, "
                         f"about {elapsed / done * (total - done):.0f}s left")

        games, training_games = project_week(table, season, week, num_runs, progress)
    logging.info(f"Model: ridge regression on pre-game team ratings (yards, turnovers, first downs, points, plays; "
                 f"offense and defense, opponent-adjusted, recency-weighted), Elo, rest and quarterback features; "
                 f"{training_games} training games, {num_runs:,} bootstrap fits")

    logging.info("\n=== FINAL PROJECTIONS ===")
    logging.info(PROJECTION_HEADER)
    final_rows = []
    for index, game in games.iterrows():
        label = week_label(game['game_type'], game['week'])
        home, away = TEAM_FULL_NAMES.get(game['home_team'], game['home_team']), TEAM_FULL_NAMES.get(game['away_team'], game['away_team'])
        actual = 0 if np.isnan(game['result']) else int(game['result'])
        mean = round(game['mean'], 1)
        logging.info(f"{index}|{label}|{season}|{home}|{away}|{mean}|{sheet.fair_line_as_home(mean)}|{round(game['std'], 2)}|{actual}")
        final_rows.append({'week': label, 'year': season, 'home_team': home, 'away_team': away, 'projected': mean, 'actual': actual})
    logging.info("Projections complete!")

    hit_rates, teaser_rates = None, None
    # The underdog signal and the two pick rules (straight underdog picks and Wong teaser legs); see nfl_signal.py
    try:
        import nfl_signal
        with run.step(3, "Building play-by-play team ratings for the pick model"):
            lab_table = lab.build_table(lab.ALL_PAIRS, refresh=True)
        with run.step(4, "Backtesting the pick model and teaser legs on every season since 1999"):
            hit_rates = nfl_signal.backtest_hit_rates(lab_table, season)
            logging.info("Backtest cover rate of the model's side by correction toward the underdog, weeks 5-16, 1999 to date "
                         "(-99 = the correction favors the favorite): "
                         + ", ".join(f"{start:g}+: {rate:.1%} ({games})" for start, rate, games in hit_rates if rate is not None))
            teaser_rates = nfl_signal.teaser_leg_rates(lab_table)
            logging.info("Wong teaser legs, weeks 5-16: " + "; ".join(
                f"{label}: {rate:.1%} of {n} since 1999, {recent:.1%} of {m} since 2018" for label, rate, n, recent, m in teaser_rates))
        with run.step(5, "Finding this week's picks and teaser legs (model corrections, reasons and DraftKings spreads)"):
            corrections = nfl_signal.week_corrections(season, week, lab_table)
            try:
                import nfl_reasons
                reasons = nfl_reasons.season_reasons(lab_table, season, range(1, week + 1))
            except Exception as e:
                logging.warning(f"Could not build the reasons for the model's leans (the picks are unaffected): {e}")
                reasons = {}
            for row in final_rows:
                row['correction'] = corrections.get((row['home_team'], row['away_team']))
                found = reasons.get((week, row['home_team'], row['away_team']))
                if found:
                    row['reason'] = found[1]
            spreads = sheet.fetch_draftkings_spreads(season, week)
            in_pick_weeks = nfl_signal.PICK_WEEKS[0] <= week <= nfl_signal.PICK_WEEKS[1]
            logging.info(f"\n=== PICKS: every underdog the correction favors, weeks {nfl_signal.PICK_WEEKS[0]}-{nfl_signal.PICK_WEEKS[1]} "
                         f"(about 55% in every era tested; 60% is not supported) ===")
            details = {}
            for row in final_rows:
                line, correction = spreads.get((row['home_team'], row['away_team'])), row.get('correction')
                if line is None or correction is None or line == 0:
                    continue
                home_is_dog = line > 0
                team, points = (row['home_team'], line) if home_is_dog else (row['away_team'], -line)
                details[(row['home_team'], row['away_team'])] = (team, points, correction if home_is_dog else -correction)
            picks = [key for key, (team, points, toward_dog) in details.items() if nfl_signal.is_pick(toward_dog, week)]
            picks.sort(key=lambda key: -details[key][2])
            for key in picks:
                team, points, toward_dog = details[key]
                logging.info(f"{team} +{points:g} ({key[1]} at {key[0]}), correction {toward_dog:.1f} pts toward the underdog")
                found = reasons.get((week, key[0], key[1]))
                if found:
                    logging.info("    Why: " + found[1].replace('\n', '\n         '))
            if not in_pick_weeks:
                logging.info(f"No picks: week {week} is outside weeks {nfl_signal.PICK_WEEKS[0]}-{nfl_signal.PICK_WEEKS[1]}")
            logging.info(f"{len(picks)} picks this week. The sheet's fair line is the spread plus the correction; "
                         f"the stats-only projections above had no edge against the spread in testing")
            logging.info("\n=== TEASER LEGS: 6-point Wong teasers through 3 and 7 (pair two legs in one teaser) ===")
            legs = 0
            for row in final_rows:
                line = spreads.get((row['home_team'], row['away_team']))
                leg = nfl_signal.teaser_leg(line, week) if line is not None else None
                if leg:
                    side, teased = leg
                    logging.info(f"{row['home_team'] if side == 'home' else row['away_team']} {teased:+g} "
                                 f"({row['away_team']} at {row['home_team']}, spread {abs(line):g})")
                    legs += 1
            logging.info(f"{legs} teaser legs this week")
        # earlier weeks already in the sheet that predate the signal get their (pre-game) corrections and reasons filled in
        with run.step(6, f"Filling in model corrections and reasons for weeks 1-{week - 1}"):
            for earlier in range(1, week):
                for (home, away), correction in nfl_signal.week_corrections(season, earlier, lab_table).items():
                    found = reasons.get((earlier, home, away))
                    final_rows.append({'week': str(earlier), 'year': season, 'home_team': home, 'away_team': away,
                                       'projected': 0.0, 'actual': None, 'correction': correction, 'backfill': True,
                                       'reason': found[1] if found else None})
    except Exception as e:
        logging.warning(f"Could not compute the underdog signal: {e}")

    with run.step(7, "Saving a snapshot of current lines and betting splits"):
        try:
            import nfl_market_moves
            logged = nfl_market_moves.append_line_snapshot(season, week)
            logging.info(f"Logged {logged} line snapshots to the line log")
        except Exception as e:
            logging.warning(f"Could not log line snapshots: {e}")

    played = table[(table['season'] == season) & table['result'].notna()]
    actual_lookup = {
        (week_label(g.game_type, g.week), season, TEAM_FULL_NAMES.get(g.home_team, g.home_team), TEAM_FULL_NAMES.get(g.away_team, g.away_team)): int(g.result)
        for g in played.itertuples()
    }
    # nflverse's spread_line is positive when the home team is favored; the sheet's spread is negative then
    closing_lookup = {
        (week_label(g.game_type, g.week), season, TEAM_FULL_NAMES.get(g.home_team, g.home_team), TEAM_FULL_NAMES.get(g.away_team, g.away_team)): -float(g.spread_line)
        for g in played.itertuples() if not np.isnan(g.spread_line)
    }
    with run.step(8, "Updating the spreadsheet (spreads, results, QB flags, formulas)"):
        try:
            output_path, rows_added, actuals_updated, spreads_filled = sheet.write_projections_spreadsheet(
                final_rows, season, actual_lookup=actual_lookup, spread_fetcher=sheet.fetch_draftkings_spreads,
                qb_flag_fetcher=sheet.fetch_qb_injury_flags, hit_rate_table=hit_rates, teaser_table=teaser_rates,
                closing_lookup=closing_lookup, current_week=final_rows[0]['week'],
            )
            logging.info(
                f"Projections spreadsheet updated: {output_path} ({rows_added} games added, "
                f"{actuals_updated} actual differences updated, {spreads_filled} DraftKings spreads filled in)"
            )
        except Exception as e:
            logging.error(f"Could not write projections spreadsheet: {e}")
    logging.info(f"Run complete in {run.elapsed():.0f}s")


if __name__ == "__main__":
    # Configuration
    PREDICT_YEAR = None  # None = current season; set a year to override
    PREDICT_WEEK = None  # None = first week with unplayed games; set a week number to override
    NUM_RUNS = BOOTSTRAP_RUNS

    main(num_runs=NUM_RUNS, predict_year=PREDICT_YEAR, predict_week=PREDICT_WEEK)
