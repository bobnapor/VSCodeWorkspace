"""The underdog signal and the two pick rules the weekly sheet flags.

Model: a ridge regression of the cover margin (result minus closing spread) on pre-game team ratings (31 play-by-play and
6 box-score stat pairs, offense and defense) plus neutral site, rest and Elo, strongly regularized and trained on the 14
seasons before the target season.  Its output, the correction, is how many points it thinks the spread is off (positive =
the home side covers).

Straight picks: every underdog the correction favors, weeks 5-16.  Validated in nfl_research_validate.py, including the
1999-2005 seasons that no research step ever scored: those underdogs covered 53.7% (1999-2005), 55.4% (2006-2017) and
55.0% (2018-2026), pushes excluded, about 4-5 picks a week; underdogs that the correction lets through do not get better
as the correction grows.  The favorites it favors cover about 51%.  A narrower key-number rule that looked like 65.8% on
2006-2026 went 53-50 on the untouched seasons and was dropped.  The correction's information (correlation about 0.07 with
the cover margin) supports about 55%; 60% would need a signal roughly twice as informative.

Teaser legs: the published Wong rule (6-point teasers through 3 and 7): underdogs of +1.5/+2.5 teased to +7.5/+8.5 and
favorites of -7.5/-8.5 teased to -1.5/-2.5.  Legs won 74-76% in every era since 1999, weeks 5-16 (favorite legs only
71% since 2018); a 2-team teaser needs 72.4% per leg at -110 and 73.9% at -120, and wins about 55-58% as one bet.
"""

import logging

import numpy as np
import pandas as pd

import nfl_lab as lab
from nfl_injury_features import TEAM_FULL_NAMES

SIGNAL_ALPHA = 10000.0
SIGNAL_FEATURES = lab.PBP_FEATURES + lab.BOX_FEATURES + lab.BASIC_CONTEXT
SIGNAL_THRESHOLD = 0.0    # any positive correction toward the underdog makes a pick
PICK_WEEKS = (5, 16)      # picks are made in these regular-season weeks only; early-season ratings are too thin
BACKWARD_TRAIN = (2006, 2019)   # the untouched seasons are scored by a model fitted on these
BACKWARD_TEST = (1999, 2005)
TEASER_POINTS = 6.0
TEASER_DOG_LINES = (1.5, 2.5)   # underdog legs, teased up through 3 and 7
TEASER_FAV_LINES = (7.5, 8.5)   # favorite legs, teased down through 7 and 3


HIT_RATE_BIN_STARTS = (-99.0, 0.0, 0.25, 0.5, 1.0)   # correction toward the underdog, in points; negative = toward the favorite


def is_pick(toward_underdog, week, min_correction=SIGNAL_THRESHOLD):
    """Whether a game is a straight pick: weeks 5-16 and the correction leans to the underdog (by at least `min_correction`)."""
    return (PICK_WEEKS[0] <= week <= PICK_WEEKS[1] and toward_underdog is not None
            and toward_underdog > 0 and toward_underdog >= min_correction)


def teaser_leg(home_spread, week):
    """The Wong teaser leg in a game as (side, teased spread for that side), or None.

    `home_spread` is the home team's spread (negative when the home team is favored).  ('home', 8.5) means tease the
    home underdog from +2.5 to +8.5; ('away', -1.5) means tease the away favorite from -7.5 to -1.5.
    """
    if home_spread is None or not PICK_WEEKS[0] <= week <= PICK_WEEKS[1]:
        return None
    line, home_is_dog = abs(home_spread), home_spread > 0
    if line in TEASER_DOG_LINES:
        return ('home' if home_is_dog else 'away', line + TEASER_POINTS)
    if line in TEASER_FAV_LINES:
        return ('away' if home_is_dog else 'home', -(line - TEASER_POINTS))
    return None


def teaser_leg_rates(table, recent_from=2018):
    """Win rate of each Wong teaser leg type, weeks 5-16, since 1999 and since `recent_from`.

    Returns [(label, rate since 1999, legs, rate since recent_from, legs)].  Teased lines are half points, so no pushes.
    """
    g = table[(table['game_type'] == 'REG') & table['result'].notna() & table['spread_line'].notna() & table['week'].between(*PICK_WEEKS)]
    line = g['spread_line'].abs()
    fav_margin = np.where(g['spread_line'] > 0, g['result'], -g['result'])
    dog_leg, fav_leg = line.isin(TEASER_DOG_LINES), line.isin(TEASER_FAV_LINES)
    win = pd.Series(np.where(fav_leg, fav_margin > line - TEASER_POINTS, fav_margin < line + TEASER_POINTS), index=g.index)
    recent = g['season'] >= recent_from
    rows = []
    for label, mask in (('Underdog +1.5/+2.5 to +7.5/+8.5', dog_leg), ('Favorite -7.5/-8.5 to -1.5/-2.5', fav_leg)):
        rows.append((label, float(win[mask].mean()), int(mask.sum()), float(win[mask & recent].mean()), int((mask & recent).sum())))
    return rows


def scored_history(table, season):
    """Every regular-season game from 1999 through `season` with the correction a model fitted without it would give.

    2006 on is walk-forward (each season fitted on the 14 before it, as in production); 1999-2005 is scored by a model
    fitted on BACKWARD_TRAIN, so those seasons are out of sample too.
    """
    fit = lab.ridge_margin(SIGNAL_FEATURES, alpha=SIGNAL_ALPHA, target='resid')
    forward = lab.walk_forward(table, fit, last_test=season)
    played = table[table['result'].notna() & (table['game_type'] == 'REG')]
    train = played[played['season'].between(*BACKWARD_TRAIN)]
    backward = []
    for past in range(BACKWARD_TEST[0], BACKWARD_TEST[1] + 1):
        test = lab.test_games(table, past)
        if not test.empty:
            scored = test.copy()
            scored['pred'] = np.asarray(fit(train, test), dtype=float)
            backward.append(scored)
    return pd.concat(backward + [forward], ignore_index=True)


def backtest_hit_rates(table, season):
    """How often the correction's side covered, by size of the correction toward the underdog, weeks PICK_WEEKS.

    Scored on every season from 1999 through the games already played in `season`, each by a model that never saw it
    (see scored_history).  Returns [(bin start, cover rate, games)], one row per bin of HIT_RATE_BIN_STARTS; the first
    bin (correction toward the favorite) is the favorite side's record, the rest the underdog side's.
    """
    frame = scored_history(table, season)
    frame = frame[(frame['resid'] != 0) & (frame['spread_line'] != 0) & frame['week'].between(*PICK_WEEKS) & (frame['game_type'] == 'REG')]
    pred = frame['pred'].to_numpy()
    win = np.where(pred > 0, frame['resid'].to_numpy() > 0, frame['resid'].to_numpy() < 0)
    toward_underdog = np.where((pred > 0) == (frame['spread_line'].to_numpy() < 0), np.abs(pred), -np.abs(pred))
    rows = []
    for i, start in enumerate(HIT_RATE_BIN_STARTS):
        end = HIT_RATE_BIN_STARTS[i + 1] if i + 1 < len(HIT_RATE_BIN_STARTS) else np.inf
        in_bin = (toward_underdog >= start) & (toward_underdog < end)
        rows.append((start, float(win[in_bin].mean()) if in_bin.any() else None, int(in_bin.sum())))
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
