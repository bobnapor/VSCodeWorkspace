"""Explain a pick: which team ratings pushed the model's correction toward a team, by how much, and how steady each effect is.

The correction model (nfl_signal.py) is a ridge regression, so a game's correction is exactly the intercept plus one
contribution per feature (coefficient x how unusual the matchup is on that rating).  This prints the largest
contributions in plain words, with both teams' ratings, and marks each effect "steady" when it points the same way when
the model is fitted on the earlier and the later half of its training seasons separately, or "flips" when it does not.

Run it from this folder after the weekly run (it uses the cached data):
    python nfl_explain.py TEN             the team's game in the next unplayed week of the current season
    python nfl_explain.py TEN 2026 5      a specific season and week
"""

import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import nfl_lab as lab
import nfl_signal
from nfl_injury_features import TEAM_FULL_NAMES

STAT_LABELS = {
    'epa_pp': 'EPA per play', 'epa_pass': 'EPA per dropback', 'epa_rush': 'EPA per run', 'success': 'success rate',
    'success_pass': 'pass success rate', 'success_rush': 'rushing success rate', 'early_epa': 'early-down EPA',
    'early_success': 'early-down success rate', 'late_epa': 'third/fourth-down EPA', 'comp_epa': 'EPA while the game is competitive',
    'comp_success': 'success rate while the game is competitive', 'explosive': 'explosive-play rate', 'sack_rate': 'sack rate',
    'int_rate': 'interception rate', 'fumble_rate': 'fumble rate', 'fumbles_lost': 'fumbles lost per game',
    'third_rate': 'third-down conversion rate', 'first_downs': 'first downs per game', 'ypa': 'yards per pass attempt',
    'ypc': 'yards per carry', 'air_yards': 'air yards per attempt', 'cpoe': 'completion % over expected', 'pass_rate': 'pass rate',
    'pass_oe': 'pass rate over expected', 'td_per_drive': 'touchdowns per drive', 'drives': 'drives per game',
    'penalties': 'penalties per game', 'penalty_yards': 'penalty yards per game', 'st_epa_for': 'special-teams EPA',
    'st_epa_against': 'special-teams EPA (opponents)', 'plays': 'plays per game', 'fd': 'first downs per game',
    'yds': 'net yards per game', 'to': 'turnovers per game', 'pts': 'points per game',
}
CONTEXT_LABELS = {'neutral': 'neutral site', 'rest_diff': 'rest days (home minus away)', 'elo_diff': 'Elo rating (home minus away, points)'}


def _fit(frame, columns):
    fill = frame[columns].median()
    scaler = StandardScaler().fit(frame[columns].fillna(fill))
    model = Ridge(alpha=nfl_signal.SIGNAL_ALPHA).fit(scaler.transform(frame[columns].fillna(fill)), frame['resid'])
    return fill, scaler, model


RATE_STATS = {'success', 'success_pass', 'success_rush', 'early_success', 'comp_success', 'explosive', 'sack_rate', 'int_rate',
              'fumble_rate', 'third_rate', 'pass_rate'}


def _describe(feature, game):
    """Plain-words name of a feature and the two teams' ratings behind it.

    An offense feature is the team's own number; a defense feature is what its opponents did against it (so for a
    defense, a higher success rate is worse and a higher interception rate is better)."""
    if feature in CONTEXT_LABELS:
        return CONTEXT_LABELS[feature], '%+.1f' % game[feature]
    side, stat = feature.split('__', 1)
    own, allowed = next(p for p in lab.ALL_PAIRS if p[0] == stat)
    base = stat[2:] if stat.startswith('o_') else stat
    name = STAT_LABELS.get(base, base)
    column = own if side == 'off' else allowed
    home, away = game['h_r_' + column], game['a_r_' + column]
    fmt = (lambda v: '%.1f%%' % (100 * v)) if base in RATE_STATS else (lambda v: '%.2f' % v)
    who = '(offense)' if side == 'off' else 'by opponents (defense)'
    return '%s %s' % (name, who), '%s %s, %s %s' % (game['home_team'], fmt(home), game['away_team'], fmt(away))


def explain(team, season=None, week=None, top=12, table=None):
    table = table if table is not None else lab.build_table(lab.ALL_PAIRS, refresh=False)
    season = season or int(table['season'].max())
    games = table[(table['season'] == season) & (table['game_type'] == 'REG') & ((table['home_team'] == team) | (table['away_team'] == team))]
    if week is None:
        unplayed = games[games['result'].isna()]
        if unplayed.empty:
            raise ValueError(f'{team} has no unplayed regular-season game in {season}')
        week = int(unplayed['week'].min())
    game = games[games['week'] == week]
    if game.empty:
        raise ValueError(f'{team} has no game in {season} week {week}')
    game = game.iloc[0]
    sign = 1.0 if game['home_team'] == team else -1.0
    opponent = game['away_team'] if sign > 0 else game['home_team']

    train = lab.training_games(table, season)
    columns = [c for c in nfl_signal.SIGNAL_FEATURES if train[c].std() > 1e-6]
    fill, scaler, model = _fit(train, columns)
    z = (game[columns].astype(float).fillna(fill).to_numpy() - scaler.mean_) / scaler.scale_
    contrib = sign * model.coef_ * z
    correction = sign * model.intercept_ + contrib.sum()
    seasons = sorted(train['season'].unique())
    halves = [train[train['season'].isin(part)] for part in (seasons[:len(seasons) // 2], seasons[len(seasons) // 2:])]
    early, late = (_fit(h, columns)[2].coef_ for h in halves)
    steady = np.sign(early) == np.sign(late)

    line = sign * -game['spread_line']          # the team's spread: positive = getting points
    status = ('underdog' if line > 0 else 'favorite' if line < 0 else "pick'em")
    print(f"{TEAM_FULL_NAMES.get(team, team)} {line:+g} vs {TEAM_FULL_NAMES.get(opponent, opponent)}, {season} week {week} "
          f"(consensus line; {team} is the {status})")
    print(f"QBs: {game['away_qb_name']} at {game['home_qb_name']} | Elo: {opponent if game['elo_diff'] * sign < 0 else team} "
          f"better by {abs(game['elo_diff']):.1f} points")
    is_pick = line > 0 and correction > 0
    print(f"Model correction toward {team}: {correction:+.2f} points -> "
          + ('a Pick (underdog the model favors)' if is_pick else 'not a Pick'))
    print(f"Fitted on {len(train):,} games from {seasons[0]}-{seasons[-1]}; steady = same direction when fitted on "
          f"{seasons[0]}-{seasons[len(seasons) // 2 - 1]} and on {seasons[len(seasons) // 2]}-{seasons[-1]} separately\n")
    order = np.argsort(-np.abs(contrib))
    print(f"  {'toward ' + team:>12}  {'effect':<7} what it is: ratings")
    for i in order[:top]:
        name, values = _describe(columns[i], game)
        print(f"  {contrib[i]:>+12.2f}  {'steady' if steady[i] else 'flips':<7} {name}: {values}")
    rest = np.ones(len(columns), bool)
    rest[order[:top]] = False
    print(f"  {contrib[rest].sum():>+12.2f}          the other {rest.sum()} ratings combined")
    print(f"  {sign * model.intercept_:>+12.2f}          baseline (home edge the model learned)")
    print(f"\nFrom steady effects {contrib[steady].sum():+.2f}, from effects that flip between the halves {contrib[~steady].sum():+.2f}.")
    print("Effects that flip have changed direction over time, so lean on them less; historically, though, the model's picks have "
          "covered about 55% whether the underdog was a good or a terrible team (nfl_research_validate.py bad).")


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    explain(args[0].upper(), int(args[1]) if len(args) > 1 else None, int(args[2]) if len(args) > 2 else None)
