"""Plain-words reasons for the correction model's lean on each game, shared by the weekly sheet and nfl_explain.py.

The correction model (nfl_signal.py) is a ridge regression, so a game's correction is exactly the intercept plus one
contribution per rating (coefficient x how unusual the matchup is on that rating).  A reason has a headline (the team
the model leans to, by how much, and the themes behind it: special teams, pace, defense, ...), then one line per theme
with its points of lean and its main rating, giving both teams' numbers and whether the team is better or worse there,
and finally whether those effects are "steady": pointing the same way when the model is fitted on the earlier and the
later half of its training seasons.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import nfl_lab as lab
import nfl_signal
from nfl_injury_features import TEAM_FULL_NAMES

STAT_LABELS = {
    'epa_pp': 'EPA per play', 'epa_pass': 'EPA per dropback', 'epa_rush': 'EPA per run', 'success': 'success rate',
    'success_pass': 'pass success rate', 'success_rush': 'rushing success rate', 'early_epa': 'early-down EPA',
    'early_success': 'early-down success rate', 'late_epa': 'third/fourth-down EPA', 'comp_epa': 'EPA in competitive game states',
    'comp_success': 'success rate in competitive game states', 'explosive': 'explosive-play rate', 'sack_rate': 'sack rate',
    'int_rate': 'interception rate', 'fumble_rate': 'fumble rate', 'fumbles_lost': 'fumbles lost per game',
    'third_rate': 'third-down conversion rate', 'first_downs': 'first downs per game', 'ypa': 'yards per pass attempt',
    'ypc': 'yards per carry', 'air_yards': 'air yards per attempt', 'cpoe': 'completion % over expected', 'pass_rate': 'pass rate',
    'pass_oe': 'pass rate over expected', 'td_per_drive': 'touchdowns per drive', 'drives': 'drives per game',
    'penalties': 'penalties committed per game', 'penalty_yards': 'penalty yards committed per game', 'st_epa_for': 'special-teams EPA',
    'st_epa_against': "opponents' special-teams EPA", 'plays': 'plays per game', 'fd': 'first downs per game',
    'yds': 'net yards per game', 'to': 'giveaways per game', 'pts': 'points per game',
}
# A defense rating is what opponents did against the team, so some read better with their own words
DEFENSE_LABELS = {
    'int_rate': 'interception rate forced', 'fumble_rate': 'fumble rate forced', 'fumbles_lost': 'fumbles recovered per game',
    'sack_rate': 'sack rate (pass rush)', 'to': 'takeaways per game', 'penalties': 'opponent penalties per game',
    'penalty_yards': 'opponent penalty yards per game', 'drives': 'opponent drives per game', 'plays': 'opponent plays per game',
    'pass_rate': 'opponent pass rate', 'pass_oe': 'opponent pass rate over expected', 'st_epa_for': "opponents' special-teams EPA",
    'st_epa_against': 'special-teams EPA against opponents',
}
RATE_STATS = {'success', 'success_pass', 'success_rush', 'early_success', 'comp_success', 'explosive', 'sack_rate', 'int_rate',
              'fumble_rate', 'third_rate', 'pass_rate'}
CONTEXT_LABELS = {'neutral': 'neutral site', 'rest_diff': 'rest days', 'elo_diff': 'Elo rating'}
# Whether a higher number is better for the team, for its offense (a defense rating, what opponents did against the
# team, reverses it); None for style and pace, which are neither
HIGHER_IS_BETTER = {
    'epa_pp': True, 'epa_pass': True, 'epa_rush': True, 'success': True, 'success_pass': True, 'success_rush': True,
    'early_epa': True, 'early_success': True, 'late_epa': True, 'comp_epa': True, 'comp_success': True, 'explosive': True,
    'sack_rate': False, 'int_rate': False, 'fumble_rate': False, 'fumbles_lost': False, 'third_rate': True,
    'first_downs': True, 'fd': True, 'ypa': True, 'ypc': True, 'air_yards': None, 'cpoe': True, 'pass_rate': None,
    'pass_oe': None, 'td_per_drive': True, 'drives': None, 'plays': None, 'penalties': False, 'penalty_yards': False,
    'st_epa_for': True, 'st_epa_against': False, 'yds': True, 'to': False, 'pts': True,
    'rest_diff': True, 'elo_diff': True,
}
# The short theme a rating belongs to in a reason's headline: (for an offense rating, for a defense rating)
THEMES = {
    'st_epa_for': ('special teams', 'special teams'), 'st_epa_against': ('special teams', 'special teams'),
    'penalties': ('penalties', 'penalties'), 'penalty_yards': ('penalties', 'penalties'),
    'drives': ('pace', 'pace'), 'plays': ('pace', 'pace'),
    'pass_rate': ('play calling', 'play calling'), 'pass_oe': ('play calling', 'play calling'),
    'int_rate': ('giveaways', 'takeaways'), 'fumble_rate': ('giveaways', 'takeaways'),
    'fumbles_lost': ('giveaways', 'takeaways'), 'to': ('giveaways', 'takeaways'),
    'sack_rate': ('sacks taken', 'pass rush'),
    'epa_pass': ('passing', 'pass defense'), 'success_pass': ('passing', 'pass defense'), 'ypa': ('passing', 'pass defense'),
    'air_yards': ('passing', 'pass defense'), 'cpoe': ('passing', 'pass defense'),
    'epa_rush': ('running game', 'run defense'), 'success_rush': ('running game', 'run defense'), 'ypc': ('running game', 'run defense'),
}
CONTEXT_THEMES = {'neutral': 'neutral site', 'rest_diff': 'rest', 'elo_diff': 'Elo rating'}


def nickname(team_code):
    return TEAM_FULL_NAMES.get(team_code, team_code).split()[-1]


def _side_and_stat(feature):
    """('off' or 'def', base stat) for a rating feature; (None, feature) for a context feature."""
    if feature in CONTEXT_LABELS:
        return None, feature
    side, stat = feature.split('__', 1)
    return side, stat[2:] if stat.startswith('o_') else stat


def feature_label(feature):
    """Plain words for a feature: 'success rate in competitive game states allowed', 'drives per game', ..."""
    side, base = _side_and_stat(feature)
    if side is None:
        return CONTEXT_LABELS[feature]
    if side == 'off':
        return STAT_LABELS.get(base, base)
    return DEFENSE_LABELS.get(base, STAT_LABELS.get(base, base) + ' allowed')


def theme(feature):
    side, base = _side_and_stat(feature)
    if side is None:
        return CONTEXT_THEMES[feature]
    offense, defense = THEMES.get(base, ('offense', 'defense'))
    return offense if side == 'off' else defense


def higher_is_better(feature):
    side, base = _side_and_stat(feature)
    better = HIGHER_IS_BETTER.get(base)
    return better if better is None or side != 'def' else not better


def _rating_columns(feature):
    side, stat = feature.split('__', 1)
    own, allowed = next(p for p in lab.ALL_PAIRS if p[0] == stat)
    column = own if side == 'off' else allowed
    return 'h_r_' + column, 'a_r_' + column


def feature_values(feature, game):
    """(home value, away value) as display text for the ratings behind a feature, or None for context features."""
    side, base = _side_and_stat(feature)
    if side is None:
        return None
    home_column, away_column = _rating_columns(feature)
    home, away = game[home_column], game[away_column]
    if base in RATE_STATS:
        small = max(abs(home), abs(away)) < 0.1
        fmt = (lambda v: '%.1f%%' % (100 * v)) if small else (lambda v: '%.0f%%' % (100 * v))
    elif 'epa' in base:
        fmt = lambda v: '%+.2f' % v
    elif base == 'cpoe':
        fmt = lambda v: '%+.1f' % v
    else:
        fmt = lambda v: '%.1f' % v
    return tuple('n/a' if pd.isna(v) else fmt(v) for v in (home, away))


def team_is_better(feature, game, sign):
    """True if the team (sign +1 home, -1 away) is better than its opponent on the feature, False if worse, None if the
    feature has no better direction or the two are level."""
    better = higher_is_better(feature)
    if better is None:
        return None
    if feature in CONTEXT_LABELS:
        gap = sign * float(game[feature])
    else:
        home_column, away_column = _rating_columns(feature)
        gap = sign * (float(game[home_column]) - float(game[away_column]))
    if gap == 0 or np.isnan(gap):
        return None
    return (gap > 0) == better


# A team's special-teams EPA against opponents is its opponent's special-teams EPA for, so these feature pairs carry the
# same numbers; a reason shows each such effect once, with the two contributions added
SAME_QUANTITY = {'def__o_st_epa_against': 'off__o_st_epa_for', 'def__o_st_epa_for': 'off__o_st_epa_against'}


def effects_toward(fitted, game, sign):
    """({feature: points toward the team}, {feature: steady}) for the team (sign +1 home, -1 away).  Features that carry
    the same numbers (SAME_QUANTITY) or the same words (box-score and play-by-play first downs or plays per game) are
    merged into the first of them, their contributions added."""
    _intercept, contrib = decompose(fitted, game)
    effects, steady, merged_into = {}, {}, {}
    for i, column in enumerate(fitted['columns']):
        feature = SAME_QUANTITY.get(column, column)
        key = merged_into.setdefault(feature_label(feature), feature)
        effects[key] = effects.get(key, 0.0) + sign * contrib[i]
        steady[key] = steady.get(key, True) and bool(fitted['steady'][i])
    return effects, steady


def fit_season(table, season):
    """The correction model for `season` (fitted on the 14 seasons before it, as in production), plus which features
    point the same way when it is fitted on the earlier and the later half of those seasons."""
    train = lab.training_games(table, season)
    columns = [c for c in nfl_signal.SIGNAL_FEATURES if train[c].std() > 1e-6]

    def fit(frame):
        fill = frame[columns].median()
        scaler = StandardScaler().fit(frame[columns].fillna(fill))
        model = Ridge(alpha=nfl_signal.SIGNAL_ALPHA).fit(scaler.transform(frame[columns].fillna(fill)), frame['resid'])
        return fill, scaler, model

    fill, scaler, model = fit(train)
    seasons = sorted(train['season'].unique())
    halves = (seasons[:len(seasons) // 2], seasons[len(seasons) // 2:])
    early, late = (fit(train[train['season'].isin(h)])[2].coef_ for h in halves)
    return {'columns': columns, 'fill': fill, 'scaler': scaler, 'model': model, 'steady': np.sign(early) == np.sign(late),
            'train_games': len(train), 'seasons': seasons, 'halves': halves}


def decompose(fitted, game):
    """(intercept, per-feature contributions) of a game's correction, both in points toward the home team."""
    columns = fitted['columns']
    x = game[columns].astype(float).fillna(fitted['fill']).to_numpy()
    z = (x - fitted['scaler'].mean_) / fitted['scaler'].scale_
    return float(fitted['model'].intercept_), fitted['model'].coef_ * z


def reason_parts(fitted, game, top_for=3, top_against=1, minimum=0.05):
    """(correction toward home, headline, detail lines, steadiness) for a game.

    Every rating's effect is grouped into a theme (special teams, pace, defense, takeaways, ...), since related ratings
    share the signal between them.
    headline:  'Titans by 0.7: defense (worse), pace; against: takeaways' - the team the model leans to, by how much,
               and the themes that pushed it there, strongest first, then the one that pushed hardest against it.
               '(worse)' marks a theme that helps the team although it is worse there (and '(better)' one that hurts it
               although it is better): the line has historically overrated that gap.
    details:   one line per headline theme: its effect in points and its main rating with both teams' numbers, e.g.
               '+0.38  defense, mainly success rate allowed: TEN 46%, HOU 39% (TEN worse)'.
    steadiness: whether the effects point the same way in the earlier and the later half of the training seasons.
    """
    intercept, contrib = decompose(fitted, game)
    correction = intercept + contrib.sum()
    sign = 1.0 if correction >= 0 else -1.0
    lean, other = (game['home_team'], game['away_team']) if sign > 0 else (game['away_team'], game['home_team'])
    effects, steady = effects_toward(fitted, game, sign)

    themes, members = {}, {}
    for feature, effect in effects.items():
        themes[theme(feature)] = themes.get(theme(feature), 0.0) + effect
        members.setdefault(theme(feature), []).append(feature)

    def main_rating(name):
        helps = themes[name] > 0
        return max((f for f in members[name] if (effects[f] > 0) == helps), key=lambda f: abs(effects[f]))

    def better(feature):
        return None if feature in CONTEXT_LABELS else team_is_better(feature, game, sign)

    def marker(name):
        is_better, helps = better(main_rating(name)), themes[name] > 0
        return '' if is_better is None or is_better == helps else ' (worse)' if helps else ' (better)'

    def rating_text(feature):
        if feature == 'neutral':
            return feature_label(feature)
        if feature in CONTEXT_LABELS:
            gap = sign * float(game[feature])
            if feature == 'rest_diff':
                return 'rest: %s %.0f %s days' % (lean, abs(gap), 'more' if gap > 0 else 'fewer')
            return '%s: %s %.0f %s' % (feature_label(feature), lean, abs(gap), 'higher' if gap > 0 else 'lower')
        lean_value, other_value = feature_values(feature, game)[::int(sign)]
        text = '%s: %s %s, %s %s' % (feature_label(feature), lean, lean_value, other, other_value)
        is_better = better(feature)
        return text if is_better is None else text + ' (%s %s)' % (lean, 'better' if is_better else 'worse')

    def detail(name):
        feature = main_rating(name)
        if len(members[name]) == 1:
            return '%+.2f  %s' % (themes[name], rating_text(feature))
        return '%+.2f  %s, mainly %s' % (themes[name], name, rating_text(feature))

    pros = [t for t in sorted(themes, key=lambda t: -themes[t]) if themes[t] >= minimum][:top_for]
    cons = [t for t in sorted(themes, key=lambda t: themes[t]) if themes[t] <= -minimum][:top_against]
    size = 'under 0.1' if abs(correction) < 0.05 else '%.1f' % abs(correction)
    headline = '%s by %s: %s' % (nickname(lean), size, ', '.join(t + marker(t) for t in pros) if pros else 'many small effects')
    if cons:
        headline += '; against: ' + ', '.join(t + marker(t) for t in cons)
    details = [detail(t) for t in pros + cons]
    steady_share = sum(v for f, v in effects.items() if steady[f]) / max(abs(correction), 1e-9)
    if steady_share >= 0.7:
        steadiness = 'Mostly steady effects.'
    elif steady_share >= 0.4:
        steadiness = 'Partly on effects that have flipped over time.'
    else:
        steadiness = 'Mostly on effects that have flipped over time, so weaker.'
    return correction, headline, details, steadiness


def reason_text(fitted, game):
    """(correction toward home, reason): the headline on the first line, then one line per theme and the steadiness."""
    correction, headline, details, steadiness = reason_parts(fitted, game)
    return correction, '\n'.join([headline] + details + [steadiness])


def season_reasons(table, season, weeks):
    """{(week, home full name, away full name): (correction, reason)} for the regular-season games of `weeks`."""
    fitted = fit_season(table, season)
    games = table[(table['season'] == season) & table['week'].isin(list(weeks)) & (table['game_type'] == 'REG')]
    out = {}
    for _, game in games.iterrows():
        correction, text = reason_text(fitted, game)
        out[(int(game['week']), TEAM_FULL_NAMES.get(game['home_team'], game['home_team']),
             TEAM_FULL_NAMES.get(game['away_team'], game['away_team']))] = (correction, text)
    return out
