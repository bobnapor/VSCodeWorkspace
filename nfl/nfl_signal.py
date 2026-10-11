"""The underdog signal: a market-anchored correction to the closing spread.

The model fits how far real results land from the closing spread (the cover margin) using pre-game team
ratings on box-score and play-by-play stats, strongly regularized so it only moves where the evidence is
consistent.  Its sign says which side it thinks the spread misprices.  In a walk-forward test over 2006-2026
the underdog-side picks covered at 54.8% (51.3% for underdogs in general); the larger the correction the
better they did, and picks on favorites showed nothing.  See nfl_research_signal.py for the evidence and
its caveats.  The configuration below is frozen to match what was tested.
"""

import logging

import numpy as np

import nfl_lab as lab
from nfl_injury_features import TEAM_FULL_NAMES

SIGNAL_ALPHA = 10000.0
SIGNAL_FEATURES = lab.PBP_FEATURES + lab.BOX_FEATURES + lab.BASIC_CONTEXT
SIGNAL_THRESHOLD = 0.5   # points of correction that flag an underdog on its own
PICKS_PER_WEEK = 3       # the week's top underdogs by correction are always flagged (walk-forward: 57.6% over 21 seasons)


def weekly_picks(candidates, per_week=PICKS_PER_WEEK, threshold=SIGNAL_THRESHOLD):
    """Underdogs to bet this week: the `per_week` largest corrections toward the underdog, plus any at the threshold.

    `candidates` is a list of (label, underdog-side correction); returns the labels flagged, largest first.
    A correction toward the favorite is never flagged, so a week can have fewer than `per_week` picks.
    """
    ranked = sorted((c for c in candidates if c[1] is not None and c[1] > 0), key=lambda c: -c[1])
    return [label for i, (label, correction) in enumerate(ranked) if i < per_week or correction >= threshold]


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
