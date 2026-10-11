"""Weekly NFL projections.

Projects every game of one week with the nflverse-based model in nfl_model.py, logs the projections, and
adds them to the season spreadsheet (nfl_sheet.py). Run it from this folder:  python nfl_prod_projections.py
"""

import logging
from datetime import datetime

import numpy as np

import nfl_lab as lab
import nfl_model as nm
import nfl_sheet as sheet
from nfl_injury_features import TEAM_FULL_NAMES

# Team stats the model rates (offense and defense each): net yards, turnovers, first downs, points and plays
MODEL_METRICS = nm.PRODUCTION_METRICS
BOOTSTRAP_RUNS = 200

PROJECTION_HEADER = "Game|Week|Year|Home|Away|Mean|FairLine(Home)|StdDev|Actual"


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


def project_week(table, season, week, runs=BOOTSTRAP_RUNS):
    """Projected home margin for each game of a week: mean, median and spread over `runs` bootstrap fits."""
    games = table[(table['season'] == season) & (table['week'] == week)].sort_values('gameday').reset_index(drop=True)
    if games.empty:
        raise ValueError(f"No games found for {season} week {week}")
    train = nm.training_frame(table, season, before_week=week)
    columns = nm.feature_columns(MODEL_METRICS, use_qb=True, use_elo=True)
    draws = nm.bootstrap_projection(train, games, columns, runs)
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

    season = predict_year or sheet.get_current_season_year()
    logging.info("Loading nflverse data...")
    table = nm.load_game_table(refresh=True)
    if not (table['season'] == season).any():
        raise ValueError(f"nflverse has no {season} schedule yet")
    week = predict_week or find_next_week_to_predict(table, season)
    logging.info(f"Predicting: Year {season}, Week {week}")

    games, training_games = project_week(table, season, week, num_runs)
    logging.info(f"Model: ridge regression on pre-game team ratings (yards, turnovers, first downs, points, plays; "
                 f"offense and defense, opponent-adjusted, recency-weighted), Elo, rest and quarterback features; "
                 f"{training_games} training games, {num_runs} bootstrap fits")

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

    hit_rates = None
    # The underdog signal: the correction to the spread that the research found worth flagging
    try:
        import nfl_signal
        lab_table = lab.build_table(lab.ALL_PAIRS, refresh=True)
        hit_rates = nfl_signal.backtest_hit_rates(lab_table, season)
        logging.info("Backtest cover rate by correction toward the underdog (walk-forward, 2006 to date): "
                     + ", ".join(f"{start:g}+: {rate:.1%} ({games})" for start, rate, games in hit_rates if rate is not None))
        corrections = nfl_signal.week_corrections(season, week, lab_table)
        for row in final_rows:
            row['correction'] = corrections.get((row['home_team'], row['away_team']))
        spreads = sheet.fetch_draftkings_spreads(season, week)
        logging.info(f"\n=== UNDERDOG PICKS (top {nfl_signal.PICKS_PER_WEEK} corrections toward the underdog, plus any of at least {nfl_signal.SIGNAL_THRESHOLD} points) ===")
        candidates, details = [], {}
        for row in final_rows:
            line, correction = spreads.get((row['home_team'], row['away_team'])), row.get('correction')
            if line is None or correction is None or line == 0:
                continue
            home_is_dog = line > 0
            team, points = (row['home_team'], line) if home_is_dog else (row['away_team'], -line)
            toward_dog = correction if home_is_dog else -correction
            key = (row['home_team'], row['away_team'])
            candidates.append((key, toward_dog))
            details[key] = (team, points, toward_dog)
        picks = nfl_signal.weekly_picks(candidates) if nfl_signal.PICK_WEEKS[0] <= week <= nfl_signal.PICK_WEEKS[1] else []
        for key in picks:
            team, points, toward_dog = details[key]
            logging.info(f"{team} +{points:g} ({key[1]} at {key[0]}), correction {toward_dog:.1f} pts")
        if not picks and not nfl_signal.PICK_WEEKS[0] <= week <= nfl_signal.PICK_WEEKS[1]:
            logging.info(f"No picks: week {week} is outside weeks {nfl_signal.PICK_WEEKS[0]}-{nfl_signal.PICK_WEEKS[1]}")
        logging.info(f"{len(picks)} underdog picks this week. The sheet's fair line is the spread plus the correction; "
                     f"the stats-only projections above had no edge against the spread in testing")
        # earlier weeks already in the sheet that predate the signal get their (pre-game) corrections filled in
        for earlier in range(1, week):
            for (home, away), correction in nfl_signal.week_corrections(season, earlier, lab_table).items():
                final_rows.append({'week': str(earlier), 'year': season, 'home_team': home, 'away_team': away,
                                   'projected': 0.0, 'actual': None, 'correction': correction, 'backfill': True})
    except Exception as e:
        logging.warning(f"Could not compute the underdog signal: {e}")

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
    try:
        output_path, rows_added, actuals_updated, spreads_filled = sheet.write_projections_spreadsheet(
            final_rows, season, actual_lookup=actual_lookup, spread_fetcher=sheet.fetch_draftkings_spreads,
            qb_flag_fetcher=sheet.fetch_qb_injury_flags, hit_rate_table=hit_rates,
        )
        logging.info(
            f"Projections spreadsheet updated: {output_path} ({rows_added} games added, "
            f"{actuals_updated} actual differences updated, {spreads_filled} DraftKings spreads filled in)"
        )
    except Exception as e:
        logging.error(f"Could not write projections spreadsheet: {e}")


if __name__ == "__main__":
    # Configuration
    PREDICT_YEAR = None  # None = current season; set a year to override
    PREDICT_WEEK = None  # None = first week with unplayed games; set a week number to override
    NUM_RUNS = BOOTSTRAP_RUNS

    main(num_runs=NUM_RUNS, predict_year=PREDICT_YEAR, predict_week=PREDICT_WEEK)
