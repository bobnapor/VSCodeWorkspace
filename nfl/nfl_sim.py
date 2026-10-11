"""Monte Carlo simulation of NFL games.

Each team's scoring is simulated event by event: a number of scoring chances (drives), each of which can end in a
touchdown or a field goal, with an extra point or two-point try after every touchdown and a small safety rate.
A shared game-environment factor (pace, weather, game flow) scales both teams' chances, which makes their
scores positively correlated.  Because points come in 3s, 6s, 7s and 8s, the margin distribution has the
key numbers (3, 6, 7, 10, 14) and the pushes the real thing has, which a bell curve does not.

The simulation is centered on the market, not on ratings: each team's expected points come from the closing
spread and total, shifted by the model's correction (see nfl_signal.py).  Ratings supply the information;
the simulation supplies the shape and the uncertainty, and turns both into probabilities and expected
values for any bet at the line on offer.
"""

import numpy as np
import pandas as pd

MARKET_PAYOUT = 100.0 / 110.0       # profit per unit staked on a -110 bet
DEFAULT_PARAMS = {
    'drives': 11,            # scoring chances per team
    'gamma_shape': 30.0,     # shared game-environment factor; smaller = more variable and more correlated
    'xp_prob': 0.94,         # extra-point success
    'two_pt_rate': 0.04,     # share of touchdowns followed by a two-point try
    'two_pt_prob': 0.48,
    'safety_rate': 0.03,     # safeties per team per game
    'td_share_base': 0.77,   # share of a team's expected points that come from touchdowns, at 22.5 expected points
    'td_share_slope': 0.003, # change in that share per expected point
}


def expected_points(line, total, correction=0.0):
    """Expected home and away points from the spread (home favored positive), the total and the model's correction."""
    margin = np.asarray(line, float) + np.asarray(correction, float)
    total = np.asarray(total, float)
    return (total + margin) / 2.0, (total - margin) / 2.0


def era_params(events, season, window=3):
    """Extra-point and two-point rates from the `window` seasons before `season` (known at the time)."""
    past = events[(events['season'] < season) & (events['season'] >= season - window)]
    if past.empty:
        return {}
    tds = past['td'].sum()
    return {
        'xp_prob': float(past['xp'].sum() / max(past['xp_att'].sum(), 1)),
        'two_pt_rate': float(past['two_pt_att'].sum() / max(tds, 1)),
        'two_pt_prob': float(past['two_pt'].sum() / max(past['two_pt_att'].sum(), 1)),
    }


def _points_per_td(p):
    return 6.0 + (1 - p['two_pt_rate']) * p['xp_prob'] + p['two_pt_rate'] * 2 * p['two_pt_prob']


def simulate_scores(e_home, e_away, n_sims=10000, params=None, rng=None, shift_sd=0.0):
    """Simulated home and away points, arrays of shape (games, n_sims).

    e_home / e_away are expected points per game.  `shift_sd` adds uncertainty about the true margin (in
    points) by moving points between the teams on each simulation, which shrinks probabilities toward 50%
    when the correction itself is uncertain.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    rng = rng if rng is not None else np.random.default_rng()
    e_home, e_away = np.atleast_1d(np.asarray(e_home, float)), np.atleast_1d(np.asarray(e_away, float))
    games = len(e_home)
    env = rng.gamma(p['gamma_shape'], 1.0 / p['gamma_shape'], size=(games, n_sims))
    shift = rng.normal(0.0, shift_sd, size=(games, n_sims)) if shift_sd > 0 else 0.0
    out = []
    for sign, expected in ((1.0, e_home), (-1.0, e_away)):
        e = np.maximum(expected[:, None] + sign * shift / 2.0, 3.0) if shift_sd > 0 else expected[:, None]
        share = np.clip(p['td_share_base'] + p['td_share_slope'] * (e - 22.5), 0.5, 0.95)
        scorable = np.maximum(e - 2 * p['safety_rate'], 0.5)
        lam_td = share * scorable / _points_per_td(p)
        lam_fg = (1 - share) * scorable / 3.0
        drives = p['drives']
        p_td = np.clip(lam_td * env / drives, 0.0, 0.9)
        td = rng.binomial(drives, p_td)
        q_fg = np.clip(lam_fg * env / (drives * (1 - p_td)), 0.0, 0.9)
        fg = rng.binomial(drives - td, q_fg)
        tries = rng.binomial(td, p['two_pt_rate'])
        two_pt = rng.binomial(tries, p['two_pt_prob'])
        xp = rng.binomial(td - tries, p['xp_prob'])
        safeties = rng.poisson(p['safety_rate'], size=td.shape)
        out.append(6 * td + xp + 2 * two_pt + 3 * fg + 2 * safeties)
    return out[0], out[1]


def bet_probabilities(home, away, line, weights=None, weight_class=None):
    """Probability that the home side wins, pushes and loses against `line`, and that the home team wins outright.

    `weights` is an optional lattice correction (see fit_lattice_weights): each simulated game is re-weighted by how
    much more or less often its final margin really occurs than the simulation produces it.  It is either one array
    for every game or a 2-D array with one row per line class, with `weight_class` giving each game's row.
    """
    margin = home - away
    diff = margin - np.asarray(line, float)[:, None]
    if weights is None:
        return {
            'home_cover': (diff > 0).mean(axis=1), 'push': (diff == 0).mean(axis=1), 'away_cover': (diff < 0).mean(axis=1),
            'home_win': (margin > 0).mean(axis=1) + 0.5 * (margin == 0).mean(axis=1),
        }
    weights = np.asarray(weights)
    size = weights.shape[-1]
    index = np.clip(np.abs(margin), 0, size - 1)
    w = (weights[index] if weights.ndim == 1 else weights[np.asarray(weight_class)[:, None], index]).astype(np.float32)
    total = w.sum(axis=1)
    return {
        'home_cover': (w * (diff > 0)).sum(axis=1) / total, 'push': (w * (diff == 0)).sum(axis=1) / total,
        'away_cover': (w * (diff < 0)).sum(axis=1) / total,
        'home_win': ((w * (margin > 0)).sum(axis=1) + 0.5 * (w * (margin == 0)).sum(axis=1)) / total,
    }


MAX_MARGIN = 60


def fit_lattice_weights(sim_margins_pmf, real_margins, smooth_beyond=35, pseudo=0.5):
    """How much more (or less) often each final margin really happens than the simulation produces it.

    `sim_margins_pmf` is the simulated pmf of the home margin pooled over the training games, offset by MAX_MARGIN;
    `real_margins` are the training games' actual margins.  The weight for |margin| is pooled over both signs, and
    margins beyond `smooth_beyond` points, where there is too little data to tell, get weight 1.
    """
    real = np.bincount((np.clip(real_margins, -MAX_MARGIN, MAX_MARGIN) + MAX_MARGIN).astype(np.int32), minlength=2 * MAX_MARGIN + 1).astype(float)
    real = real / real.sum()
    weights = np.ones(MAX_MARGIN + 1)
    for m in range(0, MAX_MARGIN + 1):
        pooled_real = real[MAX_MARGIN + m] + (real[MAX_MARGIN - m] if m else 0.0)
        pooled_sim = sim_margins_pmf[MAX_MARGIN + m] + (sim_margins_pmf[MAX_MARGIN - m] if m else 0.0)
        if m <= smooth_beyond and pooled_sim > 1e-6:
            n_real = pooled_real * len(real_margins)
            weights[m] = (n_real + pseudo) / (pooled_sim * len(real_margins) + pseudo)
    return weights


def expected_value(p_win, p_loss, payout=MARKET_PAYOUT):
    """Expected profit per unit staked; pushes return the stake."""
    return p_win * payout - p_loss


def kelly_fraction(p_win, p_loss, payout=MARKET_PAYOUT):
    """Full-Kelly stake as a share of bankroll for a bet that pays `payout` per unit, with pushes refunded."""
    return np.maximum((p_win * payout - p_loss) / payout, 0.0)


def simulate_chunked(e_home, e_away, line, n_sims=10000, params=None, rng=None, shift_sd=0.0, chunk=150, weights=None, weight_class=None):
    """Bet probabilities for many games without holding every simulated score in memory."""
    rng = rng if rng is not None else np.random.default_rng(0)
    parts = []
    for start in range(0, len(e_home), chunk):
        sl = slice(start, start + chunk)
        h, a = simulate_scores(e_home[sl], e_away[sl], n_sims, params, rng, shift_sd)
        classes = None if weight_class is None else np.asarray(weight_class)[sl]
        parts.append(pd.DataFrame(bet_probabilities(h, a, np.asarray(line)[sl], weights, classes)))
    return pd.concat(parts, ignore_index=True)


def line_class(line):
    """0 for most lines, 1 when the spread sits exactly on 3 or 7, where outcomes cluster the most."""
    return np.isin(np.abs(np.asarray(line, float)), (3.0, 7.0)).astype(int)


CALIBRATED_PARAMS = {'drives': 12, 'gamma_shape': 150.0}   # shape parameters tuned on 2008-2016 games
LATTICE_WINDOW = 12                                        # seasons of results used to fit the lattice weights


def lattice_weights_for_season(games, events, season, cache_dir=None, n_sims=500):
    """Lattice weights for `season`, fitted on the previous LATTICE_WINDOW seasons of games (cached on disk).

    `games` needs season, spread_line, total_line, home_score and away_score.
    """
    import os

    path = os.path.join(cache_dir, f'lattice_weights_{season}.npy') if cache_dir else None
    if path and os.path.exists(path):
        return np.load(path)
    train = games[games['season'].between(season - LATTICE_WINDOW, season - 1) & games['spread_line'].notna() & games['total_line'].notna()
                  & games['home_score'].notna()]
    e_home, e_away = expected_points(train['spread_line'], train['total_line'])
    params = {**CALIBRATED_PARAMS, **era_params(events, season)}
    pmf = margin_pmf(e_home, e_away, n_sims, params=params, rng=np.random.default_rng(season))
    weights = fit_lattice_weights(pmf, (train['home_score'] - train['away_score']).to_numpy())
    if path:
        np.save(path, weights)
    return weights


def margin_pmf(e_home, e_away, n_sims=1500, params_by_game=None, params=None, rng=None, chunk=150):
    """Pooled simulated pmf of the home margin over many games (offset by MAX_MARGIN); used to fit lattice weights."""
    rng = rng if rng is not None else np.random.default_rng(0)
    hist = np.zeros(2 * MAX_MARGIN + 1)
    for start in range(0, len(e_home), chunk):
        sl = slice(start, start + chunk)
        h, a = simulate_scores(e_home[sl], e_away[sl], n_sims, params, rng)
        m = (np.clip(h - a, -MAX_MARGIN, MAX_MARGIN) + MAX_MARGIN).astype(np.int32)
        hist += np.bincount(m.ravel(), minlength=2 * MAX_MARGIN + 1) / n_sims
    return hist / hist.sum()
