"""Explain a pick: which team ratings pushed the model's correction toward a team, by how much, and how steady each effect is.

The full version of the reason the weekly sheet shows (nfl_reasons.py): every large effect with both teams'
ratings, marked "steady" when it points the same way when the model is fitted on the earlier and the later half of its
training seasons separately, or "flips" when it does not.

Run it from this folder after the weekly run (it uses the cached data):
    python nfl_explain.py TEN             the team's game in the next unplayed week of the current season
    python nfl_explain.py TEN 2026 5      a specific season and week
"""

import sys

import numpy as np

import nfl_lab as lab
import nfl_reasons as nr
from nfl_injury_features import TEAM_FULL_NAMES


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

    fitted = nr.fit_season(table, season)
    intercept, _contrib = nr.decompose(fitted, game)
    effects, steady = nr.effects_toward(fitted, game, sign)
    correction = sign * intercept + sum(effects.values())

    line = sign * -game['spread_line']          # the team's spread: positive = getting points
    status = 'underdog' if line > 0 else 'favorite' if line < 0 else "pick'em"
    print(f"{TEAM_FULL_NAMES.get(team, team)} {line:+g} vs {TEAM_FULL_NAMES.get(opponent, opponent)}, {season} week {week} "
          f"(consensus line; {team} is the {status})")
    print(f"QBs: {game['away_qb_name']} at {game['home_qb_name']} | Elo: {opponent if game['elo_diff'] * sign < 0 else team} "
          f"better by {abs(game['elo_diff']):.1f} points")
    print(f"Model correction toward {team}: {correction:+.2f} points -> "
          + ('a Pick (underdog the model favors)' if line > 0 and correction > 0 else 'not a Pick'))
    seasons, (early, late) = fitted['seasons'], fitted['halves']
    print(f"Fitted on {fitted['train_games']:,} games from {seasons[0]}-{seasons[-1]}; steady = same direction when fitted on "
          f"{early[0]}-{early[-1]} and on {late[0]}-{late[-1]} separately\n")
    order = sorted(effects, key=lambda f: -abs(effects[f]))
    print(f"  {'toward ' + team:>12}  {'effect':<7} what it is: ratings")
    for feature in order[:top]:
        values = nr.feature_values(feature, game)
        shown = '' if values is None else ': %s %s, %s %s' % (game['home_team'], values[0], game['away_team'], values[1])
        better = None if feature in nr.CONTEXT_LABELS else nr.team_is_better(feature, game, sign)
        if better is not None:
            shown += ' (%s %s)' % (team, 'better' if better else 'worse')
        print(f"  {effects[feature]:>+12.2f}  {'steady' if steady[feature] else 'flips':<7} {nr.feature_label(feature)}{shown}")
    rest = sum(effects[f] for f in order[top:])
    print(f"  {rest:>+12.2f}          the other {len(order) - top} ratings combined")
    print(f"  {sign * intercept:>+12.2f}          baseline (home edge the model learned)")
    steady_part = sum(v for f, v in effects.items() if steady[f])
    print(f"\nFrom steady effects {steady_part:+.2f}, from effects that flip between the halves {sum(effects.values()) - steady_part:+.2f}.")
    print("Effects that flip have changed direction over time, so lean on them less; historically, though, the model's picks have "
          "covered about 55% whether the underdog was a good or a terrible team (nfl_research_validate.py bad).")
    print("'Worse' on an effect that helps the team means the line has historically overrated that gap.")
    print('\nSheet reason:\n    ' + nr.reason_text(fitted, game)[1].replace('\n', '\n    '))


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    explain(args[0].upper(), int(args[1]) if len(args) > 1 else None, int(args[2]) if len(args) > 2 else None)
