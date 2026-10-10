"""Backtest of the production model.

Replays a season week by week: each week is projected by a model trained only on games played before it,
exactly as nfl_prod_projections.py does, then compared with the real result and the closing spread.
Run it from this folder:  python nfl_prod_backtest.py
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

import nfl_model as nm
import nfl_sheet as sheet
from nfl_prod_projections import MODEL_METRICS


def backtest_season(table, season, weeks=None):
    """Out-of-sample projections for every played regular-season game of `season`."""
    columns = nm.feature_columns(MODEL_METRICS, use_qb=True, use_elo=True)
    played = table[(table['season'] == season) & (table['game_type'] == 'REG') & table['result'].notna()]
    outputs = []
    for week in sorted(played['week'].unique()):
        if weeks is not None and week not in weeks:
            continue
        train = nm.training_frame(table, season, before_week=week)
        games = played[played['week'] == week].copy()
        games['projected'] = nm.predict(nm.fit(train, columns), games)
        outputs.append(games)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def summarize(frame, label):
    error = frame['projected'] - frame['result']
    decided = frame[frame['result'] != 0]
    winner = ((decided['projected'] > 0) == (decided['result'] > 0)).mean()
    line = (f"{label}: {len(frame)} games | MAE {error.abs().mean():.2f} | RMSE {np.sqrt((error ** 2).mean()):.2f} "
            f"| winner {100 * winner:.1f}%")
    with_spread = frame[frame['spread_line'].notna()]
    if len(with_spread):
        market_error = (with_spread['spread_line'] - with_spread['result']).abs().mean()
        edge = with_spread['projected'] - with_spread['spread_line']
        resid = with_spread['result'] - with_spread['spread_line']
        keep = resid != 0
        win = np.where(edge[keep] > 0, resid[keep] > 0, resid[keep] < 0)
        line += f" | closing spread MAE {market_error:.2f} | ATS {100 * win.mean():.1f}% ({keep.sum()} games)"
    logging.info(line)


def main_backtest(test_years=None, test_weeks=None):
    """Backtest the given seasons (default: the current one) and write the per-game results to a CSV."""
    log_filename = f'C:/Users/Bobby/nfl_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()],
        force=True,
    )
    test_years = list(test_years or [sheet.get_current_season_year()])
    table = nm.load_game_table(refresh=True)

    frames = []
    for season in test_years:
        frame = backtest_season(table, season, test_weeks)
        if frame.empty:
            logging.warning(f"No played games found for {season}")
            continue
        summarize(frame, str(season))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    results = pd.concat(frames, ignore_index=True)
    if len(frames) > 1:
        summarize(results, 'all seasons')

    logging.info("By week:")
    for (season, week), g in results.groupby(['season', 'week']):
        summarize(g, f"  {season} week {week}")

    csv_filename = f'C:/Users/Bobby/nfl_backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results[['season', 'week', 'home_team', 'away_team', 'projected', 'result', 'spread_line']].to_csv(csv_filename, index=False)
    logging.info(f"Detailed results saved to: {csv_filename}")
    return results


if __name__ == "__main__":
    # Configuration
    TEST_YEARS = None  # None = current season; e.g. [2023, 2024, 2025] to test several
    TEST_WEEKS = None  # None = every week played; or a set of week numbers

    main_backtest(test_years=TEST_YEARS, test_weeks=TEST_WEEKS)
