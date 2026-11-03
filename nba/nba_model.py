"""The NBA models, as in the NFL suite: Model A projects the margin and the total from the ratings alone (a reference,
not expected to beat the market), and Model B, the correction model, predicts where the result lands against the
closing line (the cover margin for spreads, the over margin for totals): its output, the correction, is the points by
which it thinks the line is off.

Both are ridge regressions on standardized features (a blank feature counts as the training average).  Spread
features are home minus away (SPREAD_FEATURES): every rating of nba_lab.STATS, rest, back-to-backs, games in the last 3
days, travel, time zones, the arena's altitude, home court and availability.  Total features are home plus away
(TOTAL_FEATURES).  scored() scores seasons walk-forward: each season by a model fitted only on earlier ones.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import nba_lab

RATED = [s for s, _ in nba_lab.STATS]
ALLOWED = [s for s, allowed in nba_lab.STATS if allowed]
CONTEXT = ('rest', 'b2b', 'games_3d', 'travel_km', 'tz_shift')
PLAYERS = ('missing', 'returning', 'playing', 'missing_minutes')
SPREAD_FEATURES = ([f'd_{s}_own' for s in RATED] + [f'd_{s}_allowed' for s in ALLOWED] + ['home', 'h_altitude']
                   + [f'd_{c}' for c in CONTEXT + PLAYERS])
TOTAL_FEATURES = ([f's_{s}_own' for s in RATED if s != 'margin'] + [f's_{s}_allowed' for s in ALLOWED]
                  + ['home', 'h_altitude'] + [f'sum_{c}' for c in CONTEXT + PLAYERS])
FIRST_LINES_SEASON = 2014


def add_sums(table):
    """home + away for the context and availability columns (TOTAL_FEATURES' sum_*)."""
    table = table.copy()
    for column in CONTEXT + PLAYERS:
        table[f'sum_{column}'] = table[f'h_{column}'] + table[f'a_{column}']
    return table


class RidgeModel:
    """Standardized ridge regression; blank features count as the training mean."""

    def __init__(self, features, alpha):
        self.features, self.alpha = list(features), alpha

    def fit(self, frame, target):
        x = frame[self.features].astype(float)
        self.mean, self.std = x.mean(), x.std().replace(0, 1).fillna(1)
        self.mean = self.mean.fillna(0)
        z = ((x - self.mean) / self.std).fillna(0)
        y = frame[target].astype(float)
        keep = y.notna()
        self.ridge = Ridge(alpha=self.alpha).fit(z[keep].values, y[keep].values)
        return self

    def predict(self, frame):
        z = ((frame[self.features].astype(float) - self.mean) / self.std).fillna(0)
        return self.ridge.predict(z.values)

    def weights(self):
        """The fitted weights, in points per standard deviation of each feature, largest first."""
        return pd.Series(self.ridge.coef_, index=self.features).sort_values(key=np.abs, ascending=False)


def scored(table, kind, alpha, first_test, last_test, train_from=FIRST_LINES_SEASON, features=None):
    """Walk-forward predictions for `kind`: 'cover' or 'over' (Model B: the correction, needing closing lines to train)
    or 'margin' or 'points' (Model A).  `features` replaces the default feature list.  Returns the scored rows with a
    'pred' column."""
    features = features or (SPREAD_FEATURES if kind in ('cover', 'margin') else TOTAL_FEATURES)
    table = add_sums(table)

    def fit_predict(train, test):
        train = train[train[kind].notna()]
        return RidgeModel(features, alpha).fit(train, kind).predict(test)
    return nba_lab.walk_forward(table, fit_predict, first_test, last_test, train_from=train_from)
