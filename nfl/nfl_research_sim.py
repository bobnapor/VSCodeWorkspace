"""Monte Carlo evaluation of the underdog signal.

1. Calibrate the correction: a walk-forward isotonic map from the ridge's predicted correction to the cover margin
   that actually followed, then check the simulated probabilities against what happened.
2. Choose bets by simulated expected value (EV) and by the size of the correction, and compare.
3. Cash the same edge through the moneyline instead of the spread.
4. Stake by flat percentages and by fractional Kelly on the simulated probabilities.
5. Season-level Monte Carlo: a season-block bootstrap of the realized results, a Bayesian posterior for the true win
   rate, and forward simulation of seasons and bankrolls under that uncertainty.

Everything is walk-forward: the ridge correction, its calibration, the lattice weights and the extra-point rates
for a season come only from earlier seasons.  Run it from the nfl folder:  python nfl_research_sim.py
"""

import math

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import nfl_lab as lab
import nfl_sim as sim
from nfl_pbp_features import load_scoring_events

PARAMS = {'drives': 12, 'gamma_shape': 150.0}   # shape parameters tuned on 2008-2016 games
FEATURES = lab.PBP_FEATURES + lab.BOX_FEATURES + lab.BASIC_CONTEXT
FIRST_TEST = 2012
LATTICE_WINDOW = 12
N_SIMS = 4000
BREAK_EVEN = 110.0 / 210.0


def american_profit(odds):
    odds = np.asarray(odds, float)
    return np.where(odds > 0, odds / 100.0, 100.0 / np.abs(odds))


def lattice_weights(games, events, season):
    train = games[games['season'].between(season - LATTICE_WINDOW, season - 1)]
    p = {**PARAMS, **sim.era_params(events, season)}
    pmf = sim.margin_pmf(train['e_home'].to_numpy(), train['e_away'].to_numpy(), 500, params=p, rng=np.random.default_rng(season))
    return sim.fit_lattice_weights(pmf, train['margin'].to_numpy())


def calibrate_toward_underdog(history, test, bins=25):
    """Map the ridge's correction to the cover margin that followed, using earlier seasons' out-of-sample predictions.

    Works in the underdog's frame (positive = toward the underdog) because that is where the signal lives.
    Returns the calibrated correction for `test` in the home team's frame.
    """
    sign_h = np.where(history['spread_line'] < 0, 1.0, -1.0)
    d_h = history['correction'].to_numpy() * sign_h
    y_h = history['resid'].to_numpy() * sign_h
    edges = np.unique(np.quantile(d_h, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.digitize(d_h, edges[1:-1]), 0, len(edges) - 2)
    centers = np.array([d_h[which == i].mean() for i in range(len(edges) - 1)])
    means = np.array([y_h[which == i].mean() for i in range(len(edges) - 1)])
    counts = np.array([(which == i).sum() for i in range(len(edges) - 1)], float)
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(centers, means, sample_weight=counts)
    sign_t = np.where(test['spread_line'] < 0, 1.0, -1.0)
    return iso.predict(test['correction'].to_numpy() * sign_t) * sign_t, iso


def build_frame():
    """One row per game 2012-2026 with the raw and calibrated corrections and simulated outcome probabilities."""
    events = load_scoring_events(1999, 2026)
    table = lab.default_table()
    corrections = lab.walk_forward(table, lab.ridge_margin(FEATURES, alpha=10000.0, target='resid'), first_test=2006)[['game_id', 'pred']].rename(columns={'pred': 'correction'})
    games = table[table['result'].notna() & table['spread_line'].notna() & table['total_line'].notna() & (table['game_type'] == 'REG')].copy()
    games['margin'] = games['home_score'] - games['away_score']
    games['e_home'], games['e_away'] = sim.expected_points(games['spread_line'], games['total_line'])
    games = games.merge(corrections, on='game_id', how='left')
    frames, last_iso = [], None
    for season in range(FIRST_TEST, int(games['season'].max()) + 1):
        test = games[games['season'] == season].copy()
        if test.empty:
            continue
        history = games[(games['season'] < season) & games['correction'].notna()]
        test['correction_cal'], last_iso = calibrate_toward_underdog(history, test)
        weights = lattice_weights(games, events, season)
        params = {**PARAMS, **sim.era_params(events, season)}
        for tag, c in (('raw', test['correction'].to_numpy()), ('cal', test['correction_cal'].to_numpy()), ('mkt', np.zeros(len(test)))):
            e_h, e_a = sim.expected_points(test['spread_line'], test['total_line'], c)
            probs = sim.simulate_chunked(e_h, e_a, test['spread_line'].to_numpy(), N_SIMS, params, np.random.default_rng(season * 10 + len(tag)), weights=weights)
            for col in probs.columns:
                test[f'{tag}_{col}'] = probs[col].to_numpy()
        frames.append(test)
    frame = pd.concat(frames, ignore_index=True)
    frame['home_result'] = np.sign(frame['margin'] - frame['spread_line'])      # +1 home covered, -1 away covered, 0 push
    frame['dog_is_home'] = frame['spread_line'] < 0
    xs = np.array([-1.0, -0.5, 0.0, 0.3, 0.5, 1.0, 1.5])
    print('calibration map fitted for 2026 (toward-underdog correction -> expected cover margin in points): ' +
          ', '.join('%.1f -> %+.2f' % (x, y) for x, y in zip(xs, last_iso.predict(xs))))
    return frame


def spread_bets(frame, tag='cal', by='ev', min_ev=None, min_corr=0.0, dog_only=True):
    """One bet per game on the side the correction (or the simulated EV) favors, filtered."""
    corr_col = 'correction_cal' if tag == 'cal' else 'correction'
    rows = []
    for g in frame.itertuples():
        p_home, p_away = getattr(g, f'{tag}_home_cover'), getattr(g, f'{tag}_away_cover')
        c = getattr(g, corr_col)
        ev_home, ev_away = sim.expected_value(p_home, p_away), sim.expected_value(p_away, p_home)
        home = (ev_home >= ev_away) if by == 'ev' else (c > 0)
        ev = ev_home if home else ev_away
        is_dog = home == g.dog_is_home
        if g.spread_line == 0 or abs(c) < min_corr or (dog_only and not is_dog) or (min_ev is not None and ev < min_ev):
            continue
        result = g.home_result if home else -g.home_result
        p_win, p_loss = (p_home, p_away) if home else (p_away, p_home)
        rows.append({'season': g.season, 'week': g.week, 'ev': ev, 'p_win': p_win, 'p_loss': p_loss, 'is_dog': is_dog, 'result': result,
                     'profit': sim.MARKET_PAYOUT if result > 0 else (-1.0 if result < 0 else 0.0), 'corr': abs(c)})
    return pd.DataFrame(rows)


def describe(label, bets):
    if bets.empty:
        print('%-50s no bets' % label)
        return
    decided = bets[bets['result'] != 0]
    d, h = bets[bets['season'] <= 2017], bets[bets['season'] > 2017]
    by = bets.groupby('season')['profit'].sum()
    print('%-50s n=%4d win=%.3f ROI %+6.1f%% | 2012-17 %+6.1f%% (n=%d) | 2018-26 %+6.1f%% (n=%d) | losing seasons %d/%d' % (
        label, len(bets), (decided['result'] > 0).mean(), 100 * bets['profit'].mean(), 100 * d['profit'].mean() if len(d) else float('nan'), len(d),
        100 * h['profit'].mean() if len(h) else float('nan'), len(h), int((by < 0).sum()), len(by)))


def reliability(label, bets, bins=4):
    decided = bets[bets['result'] != 0].copy()
    decided['pred'] = decided['p_win'] / (decided['p_win'] + decided['p_loss'])
    decided['bin'] = pd.qcut(decided['pred'], bins, duplicates='drop')
    print(label)
    for b, g in decided.groupby('bin'):
        print('   predicted %.3f-%.3f: n=%4d predicted %.3f realized %.3f' % (b.left, b.right, len(g), g['pred'].mean(), (g['result'] > 0).mean()))
    print('   all: predicted %.3f vs realized %.3f' % (decided['pred'].mean(), (decided['result'] > 0).mean()))


def moneyline_bets(frame, tag, min_ev):
    ml = frame[frame['home_moneyline'].notna() & frame['away_moneyline'].notna()]
    rows = []
    for g in ml.itertuples():
        p = getattr(g, f'{tag}_home_win')
        pay_h, pay_a = float(american_profit(g.home_moneyline)), float(american_profit(g.away_moneyline))
        ev_h, ev_a = p * pay_h - (1 - p), (1 - p) * pay_a - p
        home = ev_h >= ev_a
        dog = (home and g.home_moneyline > 0) or ((not home) and g.away_moneyline > 0)
        if (ev_h if home else ev_a) < min_ev or not dog:
            continue
        won = g.margin > 0 if home else g.margin < 0
        tie = g.margin == 0
        rows.append({'season': g.season, 'result': 0 if tie else (1 if won else -1), 'profit': 0.0 if tie else ((pay_h if home else pay_a) if won else -1.0)})
    return pd.DataFrame(rows)


def stake_path(bets, fraction=0.0, flat=0.0, per_bet_cap=0.03, week_cap=0.15):
    bank, peak, worst = 1.0, 1.0, 0.0
    for _, g in bets.groupby(['season', 'week']):
        stakes = np.full(len(g), flat) if flat else np.minimum(fraction * sim.kelly_fraction(g['p_win'].to_numpy(), g['p_loss'].to_numpy()), per_bet_cap)
        if stakes.sum() > week_cap:
            stakes = stakes * week_cap / stakes.sum()
        bank *= 1 + float((stakes * g['profit'].to_numpy()).sum())
        peak = max(peak, bank)
        worst = max(worst, 1 - bank / peak)
    return bank, worst


def season_monte_carlo(bets, rng, n_draws=20000):
    print('\n=== 5. season-level Monte Carlo for the chosen rule (%d bets over %d seasons) ===' % (len(bets), bets['season'].nunique()))
    per_season = bets.groupby('season')['profit'].agg(['sum', 'count'])
    # (a) bootstrap over whole seasons, so the season-to-season swings are kept
    idx = rng.integers(0, len(per_season), size=(n_draws, len(per_season)))
    roi = per_season['sum'].to_numpy()[idx].sum(axis=1) / per_season['count'].to_numpy()[idx].sum(axis=1)
    print('(a) season-block bootstrap of realized results: ROI median %+.1f%%, 90%% interval [%+.1f%%, %+.1f%%], P(ROI > 0) = %.3f' % (
        100 * np.median(roi), 100 * np.percentile(roi, 5), 100 * np.percentile(roi, 95), (roi > 0).mean()))
    # (b) Bayesian posterior for the true win rate among decided bets
    decided = bets[bets['result'] != 0]
    wins, losses = int((decided['result'] > 0).sum()), int((decided['result'] < 0).sum())
    print('(b) posterior for the true win rate (%d wins, %d losses; break-even %.3f):' % (wins, losses, BREAK_EVEN))
    posteriors = {}
    for label, (a0, b0) in (('flat prior Beta(1,1)', (1, 1)), ('skeptical prior centered on 50% (worth 1,000 bets)', (500, 500)),
                            ('very skeptical prior centered on break-even (worth 2,000 bets)', (BREAK_EVEN * 2000, (1 - BREAK_EVEN) * 2000))):
        draws = rng.beta(a0 + wins, b0 + losses, size=n_draws)
        posteriors[label] = draws
        print('    %-62s mean %.3f, 90%% interval [%.3f, %.3f], P(true rate > break-even) = %.2f' % (
            label, draws.mean(), np.percentile(draws, 5), np.percentile(draws, 95), (draws > BREAK_EVEN).mean()))
    # (c) forward simulation of future seasons using the skeptical posterior
    bets_per_season = len(bets) / bets['season'].nunique()
    p_draws = posteriors['skeptical prior centered on 50% (worth 1,000 bets)']
    print('(c) simulated future seasons (%.0f bets a season, -110, flat 1 unit), true win rate drawn from the skeptical posterior:' % bets_per_season)
    n_bets = rng.poisson(bets_per_season, size=n_draws)
    won = rng.binomial(n_bets, p_draws)
    units = won * sim.MARKET_PAYOUT - (n_bets - won)
    print('    season result in units: median %+.1f, 10th percentile %+.1f, 90th percentile %+.1f; P(losing season) = %.2f; P(down 10+ units) = %.2f' % (
        np.median(units), np.percentile(units, 10), np.percentile(units, 90), (units < 0).mean(), (units <= -10).mean()))
    print('(d) sensitivity: what if the true win rate were...  (5 seasons of bets, flat 1%% of bankroll per bet, weekly slates)')
    weeks = 17
    for p_true in (0.50, 0.524, 0.54, 0.56, 0.58):
        bank = np.ones(2000)
        worst = np.zeros(2000)
        peak = np.ones(2000)
        for _ in range(5 * weeks):
            n = rng.poisson(bets_per_season / weeks, size=2000)
            wins_w = rng.binomial(n, p_true)
            bank = bank * (1 + 0.01 * (wins_w * sim.MARKET_PAYOUT - (n - wins_w)))
            peak = np.maximum(peak, bank)
            worst = np.maximum(worst, 1 - bank / peak)
        print('    true win rate %.3f: median bankroll after 5 seasons x%.2f (10th-90th pct x%.2f-x%.2f), P(ever down 15%%+ from a peak) = %.2f, P(finish below start) = %.2f' % (
            p_true, np.median(bank), np.percentile(bank, 10), np.percentile(bank, 90), (worst >= 0.15).mean(), (bank < 1).mean()))


def main():
    frame = build_frame()
    print('games %d, seasons %d-%d' % (len(frame), frame['season'].min(), frame['season'].max()))

    print('\n=== 1. does calibrating the correction fix the simulated probabilities? (underdog picks, correction >= 0.5 in the raw frame) ===')
    reliability('raw ridge correction:', spread_bets(frame, 'raw', by='corr', min_corr=0.5))
    reliability('calibrated correction (same bets chosen by the raw size):', spread_bets(frame, 'cal', by='corr', min_corr=0.0)[lambda d: d['corr'] > 0.3])

    print('\n=== 2. choosing bets: by correction size or by simulated EV (flat 1 unit at -110, underdogs only) ===')
    describe('raw correction >= 0.5, no EV filter', spread_bets(frame, 'raw', by='corr', min_corr=0.5))
    describe('raw correction >= 0.5 and simulated EV >= 0', spread_bets(frame, 'raw', by='corr', min_corr=0.5, min_ev=0.0))
    for ev in (0.0, 0.02, 0.04, 0.06, 0.08):
        describe('calibrated, simulated EV >= %.0f%%' % (100 * ev), spread_bets(frame, 'cal', min_ev=ev))
    for ev in (0.02, 0.04):
        describe('calibrated, either side, EV >= %.0f%%' % (100 * ev), spread_bets(frame, 'cal', min_ev=ev, dog_only=False))

    print('\n=== 3. the same edge through the moneyline (underdog moneylines at the closing price) ===')
    for tag in ('raw', 'cal', 'mkt'):
        for ev_min in (0.0, 0.05, 0.10):
            b = moneyline_bets(frame, tag, ev_min)
            if len(b) > 30:
                describe(f'ML underdogs, {tag} probabilities, EV >= {100 * ev_min:.0f}%', b)

    print('\n=== 4. staking the calibrated EV >= 4% underdog bets (weekly compounding) ===')
    chosen = spread_bets(frame, 'cal', min_ev=0.04)
    years = chosen['season'].nunique()
    for label, kw in (('flat 1% of bankroll per bet', {'flat': 0.01}), ('flat 2% of bankroll per bet', {'flat': 0.02}),
                      ('quarter Kelly (cap 3% per bet)', {'fraction': 0.25}), ('half Kelly (cap 3% per bet)', {'fraction': 0.5}), ('full Kelly (cap 3% per bet)', {'fraction': 1.0})):
        bank, worst = stake_path(chosen, **kw)
        print('   %-34s final bankroll x%.2f over %d seasons (%+.1f%%/yr), worst drawdown %.0f%%' % (label, bank, years, 100 * (bank ** (1 / years) - 1), 100 * worst))

    season_monte_carlo(chosen, np.random.default_rng(11))
    return frame


if __name__ == '__main__':
    main()
