"""The underdog signal: a market-anchored correction to the closing spread, and the rule that turns it into picks.

The model fits how far real results land from the closing spread (the cover margin) using pre-game team
ratings on box-score and play-by-play stats, strongly regularized so it only moves where the evidence is
consistent.  Its sign says which side it thinks the spread misprices.  The edge is only on underdogs, and
within those it concentrates on lines that sit exactly on a key number (3, 6, 7, 10, 14): in a walk-forward
test over 2006-2026, weeks 5-16, an underdog on a key line with a correction of at least 0.25 points toward it
covered 65.8% (228 picks, about 11 a season, 17 of 20 seasons above break-even); at any positive correction 60.4%
(about 17 a season).  Underdogs on other lines covered 52-55% and favorites about 51%.  The key-number cut was found
among about 25 conditions that were scanned, so it is a strong lead and not proven; see nfl_research_select.py
for the evidence and its caveats.  The configuration below is frozen to match what was tested.
"""

import logging

import numpy as np

import nfl_lab as lab
from nfl_injury_features import TEAM_FULL_NAMES
from nfl_sheet import KEY_LINES

SIGNAL_ALPHA = 10000.0
SIGNAL_FEATURES = lab.PBP_FEATURES + lab.BOX_FEATURES + lab.BASIC_CONTEXT
SIGNAL_THRESHOLD = 0.25   # points of correction toward the underdog needed before a game is picked
PICK_WEEKS = (5, 16)      # picks are made in these regular-season weeks only; early-season ratings are too thin


HIT_RATE_BIN_STARTS = (-99.0, 0.0, 0.25, 0.5, 1.0)   # correction toward the underdog, in points; negative = toward the favorite


def is_pick(line, toward_underdog, week, min_correction=SIGNAL_THRESHOLD):
    """Whether a game is worth picking: weeks 5-16, the underdog's line on a key number, and the model leaning to the underdog.

    `line` is the underdog's spread as a size in points (3 for +3), `toward_underdog` the correction toward it.
    """
    return (PICK_WEEKS[0] <= week <= PICK_WEEKS[1] and abs(line) in KEY_LINES
            and toward_underdog is not None and toward_underdog > 0 and toward_underdog >= min_correction)


def backtest_hit_rates(table, season):
    """How often the correction's side covered, by size of the correction toward the underdog, for key lines and other lines.

    Walk-forward from 2006 through the games already played in `season`, weeks PICK_WEEKS only (each season
    predicted by a model fitted only on earlier seasons, as in production).  Returns
    [(bin start, key-line cover rate, key-line games, other-line cover rate, other-line games)], one row per bin of
    HIT_RATE_BIN_STARTS, so the sheet's Backtest ATS Win % reflects every game played to date.
    """
    frame = lab.walk_forward(table, lab.ridge_margin(SIGNAL_FEATURES, alpha=SIGNAL_ALPHA, target='resid'), last_test=season)
    frame = frame[(frame['resid'] != 0) & frame['week'].between(*PICK_WEEKS)]
    pred = frame['pred'].to_numpy()
    win = np.where(pred > 0, frame['resid'].to_numpy() > 0, frame['resid'].to_numpy() < 0)
    toward_underdog = np.where((pred > 0) == (frame['spread_line'].to_numpy() < 0), np.abs(pred), -np.abs(pred))
    key = frame['spread_line'].abs().isin(KEY_LINES).to_numpy()
    rows = []
    for i, start in enumerate(HIT_RATE_BIN_STARTS):
        end = HIT_RATE_BIN_STARTS[i + 1] if i + 1 < len(HIT_RATE_BIN_STARTS) else np.inf
        in_bin = (toward_underdog >= start) & (toward_underdog < end)
        row = [start]
        for mask in (in_bin & key, in_bin & ~key):
            row += [float(win[mask].mean()) if mask.any() else None, int(mask.sum())]
        rows.append(tuple(row))
    return rows


def week_corrections(season, week, table=None):
    """Correction in points for each game of a week, positive when the model thinks the home side beats the spread.

    Returns {(home team, away team): correction} with full team names.  Fits only on seasons before `season`,
    exactly as the walk-forward test did.
    """
    table = table if table is not None else lab.build_table(lab.ALL_PAIRS, refresh=True)
    games = table[(table['season'] == season) & (table['week'] == week) & (table['game_type'] == 'REG')]
    if games.empty:
        return {}
    train = lab.training_games(table, season)
    correction = lab.ridge_margin(SIGNAL_FEATURES, alpha=SIGNAL_ALPHA, target='resid')(train, games)
    logging.info(f"Underdog signal model fitted on {len(train)} games from seasons {int(train['season'].min())}-{int(train['season'].max())}")
    return {
        (TEAM_FULL_NAMES.get(g.home_team, g.home_team), TEAM_FULL_NAMES.get(g.away_team, g.away_team)): float(c)
        for g, c in zip(games.itertuples(), np.asarray(correction))
    }
