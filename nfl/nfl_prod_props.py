"""Weekly NFL player prop picks (DraftKings), alongside the spread picks of nfl_prod_projections.py.

Two rules (nfl_prop_signal.py), each with its own forecast model (nfl_prop_model.py):
  Overs   the early-week model (only what is known when props open: no injury report, opening spreads and totals)
          gives the over 65% or more, while DraftKings' line is still at its opening number.  The market moves toward
          these overs during the week, so they must be bet early: run this as soon as the week's props are posted.
          No over for a player back from an absence (his team played 3+ games since his last: the returner skip).
  Unders  the full model (final injury report, closing game lines) gives the under 65% or more; they hold at closing
          lines, so an under is picked only within 24 hours of its game's kickoff (UNDER_WINDOW_HOURS) and once the
          game's final injury report is in nflverse's data: the daily run on game day picks them up.
          On since October 21, 2026 (SIDES below).
Picks go best first, with the worst price still worth taking, to a log file and to <Desktop>/<year>_nfl_props.xlsx
(nfl_props_sheet.py).  Earlier picks are graded from nflverse results, and the week's lines are appended to
data_cache/prop_line_log.csv so the picks' closing-line value can be measured.
The spread runner and its workbook are not touched.  Run it from this folder:  python nfl_prod_props.py
"""

import logging
import os
from datetime import datetime

import pandas as pd

import nfl_prop_lines as pl
import nfl_prop_model as pm
import nfl_prop_signal as ps
import nfl_props_sheet as props_sheet
import nfl_sheet as sheet
import nflverse_data as nd
from nfl_injury_features import TEAM_RENAMES
from nfl_prod_projections import RunLog, find_next_week_to_predict

TOTAL_STEPS = 7
DEFAULT_SIDES = ('Over', 'Under')
# How each side is picked: the model variant, the lines its backtest record is graded at, and its rule
RULES = {
    'Over': dict(early=True, graded_at='open', label='overs at opening lines', markets=ps.OVER_MARKETS,
                 picks=ps.over_picks),
    'Under': dict(early=False, graded_at='close', label='unders at closing lines', markets=ps.PICK_MARKETS,
                  picks=ps.rule_picks),
}
RULE_LINES = (
    f"Over rule: bet the Over at DraftKings' opening line when the early-week model gives it a chance of "
    f"{ps.OVER_MIN_CHANCE:.0%} or more, in every market but passing TDs and interceptions.  Run the script as soon as "
    "DraftKings posts the week's props (usually Tuesday or Wednesday) and bet right away: an over is shown only while its "
    "line is still at the opening number.",
    "Why the overs work only early: the market moves toward them during the week (2024-2025: the line rose for 53% of them "
    "and fell for 11%).  At the opener they won 59%; at the closing line about 51%.  The early-week model uses only what "
    "is known when props open: no injury report, and the opening spreads and totals.",
    f"Late overs: an over still at its opener within {ps.OVER_LATE_HOURS} hours of kickoff is almost surely a line the "
    "market never moves, and those won only about 52% in the backtest; such picks are marked LATE and priced at that "
    "rate, so their Bet At asks for about even money.",
    f"Returner skip: no over pick for a player whose team has played {ps.RETURNER_MISSED}+ games since his last game "
    f"(week {ps.RETURNER_FIRST_WEEK} on).  The models build his numbers from his last games played, however long ago, so "
    "after an absence they forecast his old role: receiving forecasts for such players ran 20-30% high, and their overs "
    "went 28-35 (44%) in the backtest.",
    f"Under rule: bet the Under when the full model (final injury report, closing game lines) gives it {ps.MIN_CHANCE:.0%} "
    "or more, in " + ', '.join(pl.MARKETS[m][1] for m in ps.PICK_MARKETS)
    + f".  It holds at closing lines, so unders are picked only within {ps.UNDER_WINDOW_HOURS} hours of their game's "
    "kickoff, once its final injury report is in nflverse's data: the daily run on game day picks them up.",
    "Left out: passing TDs and interceptions (books move the price rather than the line, so a win rate at the line says "
    "little without prices); the under rule also leaves out pass attempts and completions (no information there).",
    f"Bet At: the worst DraftKings price at which a pick still returns {ps.MIN_EDGE:.0%} on average at its Hist. Win %. "
    "The free ESPN feed has DraftKings' lines but not its prices, so check the price in the app.",
    "Forecasts: one gradient-boosted model per market on nflverse box scores, game lines, injury reports and starting "
    f"QBs, refitted each season on the {pm.TRAIN_SEASONS} seasons before it; the chances come from what happened after "
    f"similar out-of-sample forecasts in the previous {pm.DIST_SEASONS} seasons.",
    "Lines: DraftKings via ESPN's public odds API, logged at every run in data_cache/prop_line_log.csv.  Line vs Close: "
    "how many more units of the stat your line gave than the closing line (positive = beat the close).",
    "Voids: DraftKings voids a prop when the player does not play; a pick whose player has no stats in a finished game "
    "is marked Void (check it if he played and simply recorded nothing).",
)
EVIDENCE = (
    "Overs, discovery (ESPN BET 2024-2025 opening lines, early-week model): 58.9% of 1,174; receptions 65.3%, rush "
    "attempts 62.2%, pass + rush yards 60.6%, passing yards 60.3%.  By what the line did next: 67.0% when it later rose "
    "(54% of the picks), 52.1% when it never moved, 40.9% when it fell.",
    "Overs, untouched test (DraftKings 2026 weeks 1-6 opening lines): 55.7% of 183 (+11.7 units at -110); receptions "
    "63.5% of 63.  Where the close is known, the line rose for 26% of the picks and fell for 11%.",
    "Unders, discovery (ESPN BET 2024-2025 closing lines): 56.5% of 2,873; untouched test (DraftKings 2026 closing lines "
    "last moved before kickoff): 57.4% of 284.",
    "Against closing lines the full model's overs carry no information (about 50% at every threshold): the over edge is "
    "the early bet, not the model alone.",
    "Break-even win rate: 52.4% at -110, 53.5% at -115, 54.5% at -120, 56.5% at -130.",
)


def kickoff_label(gameday, gametime):
    try:
        return datetime.strptime(f'{gameday} {gametime}', '%Y-%m-%d %H:%M').strftime('%a %m/%d %I:%M %p')
    except (TypeError, ValueError):
        return str(gameday)


def upcoming_rows(lines, week_games, weeks, players, season, week):
    """One row per player to forecast this week (every player with a line, plus both starting quarterbacks)."""
    latest = weeks.sort_values(['season', 'week']).drop_duplicates('player_id', keep='last').set_index('player_id')
    roster = players.dropna(subset=['gsis_id']).drop_duplicates('gsis_id').set_index('gsis_id')
    sides = {g.game_id: (g.home_team, g.away_team) for g in week_games.itertuples()}
    wanted = set(zip(lines['gsis_id'], lines['game_id'])) if len(lines) else set()
    for g in week_games.itertuples():
        for qb in (g.home_qb_id, g.away_qb_id):
            if isinstance(qb, str):
                wanted.add((qb, g.game_id))
    rows, unplaced = [], []
    for gsis, game_id in sorted(wanted):
        home, away = sides[game_id]
        # nflverse's current roster team first: the team of a player's last game is stale after a move, and in a
        # game against his old team it would put him on the wrong side
        candidates = []
        if gsis in roster.index and isinstance(roster.at[gsis, 'latest_team'], str):
            candidates.append(TEAM_RENAMES.get(roster.at[gsis, 'latest_team'], roster.at[gsis, 'latest_team']))
        if gsis in latest.index:
            candidates.append(TEAM_RENAMES.get(latest.at[gsis, 'team'], latest.at[gsis, 'team']))
        team = next((t for t in candidates if t in (home, away)), None)
        if team is None:
            unplaced.append(gsis)
            continue
        source = latest.loc[gsis] if gsis in latest.index else roster.loc[gsis]
        name = source.get('player_display_name') if gsis in latest.index else source.get('display_name')
        rows.append({'player_id': gsis, 'player_display_name': name, 'position': source.get('position'),
                     'season': season, 'week': week, 'season_type': 'REG', 'game_id': game_id, 'team': team,
                     'opponent_team': away if team == home else home})
    if unplaced:
        logging.info(f"{len(unplaced)} players with lines could not be placed on either team of their game; skipped")
    return pd.DataFrame(rows)


def market_starters(games, upcoming, lines, season, week):
    """The schedule with each team's starting QB set to the quarterback DraftKings posts passing props for.

    nflverse's expected starter can lag injury news; a book posts passing props only for the quarterback it expects
    to start, so when that is someone else he is used as the starter (and logged).  A listed starter who has passing
    props himself is kept.
    """
    passing = lines[lines['market'].isin(['pass_yds', 'pass_att', 'pass_cmp'])] if len(lines) else lines
    if passing is None or passing.empty or upcoming.empty:
        return games
    teams = upcoming.set_index(['player_id', 'game_id'])['team'].to_dict()
    passers = set(zip(passing['gsis_id'], passing['game_id']))
    games = games.copy()
    for (gsis, game_id), name in passing.groupby(['gsis_id', 'game_id'])['player_name'].first().items():
        team = teams.get((gsis, game_id))
        row = games.index[(games['game_id'] == game_id)]
        if team is None or row.empty:
            continue
        side = 'home' if TEAM_RENAMES.get(games.at[row[0], 'home_team'], games.at[row[0], 'home_team']) == team else 'away'
        listed = games.at[row[0], f'{side}_qb_id']
        if listed != gsis and (listed, game_id) not in passers:
            logging.info(f"{team}: DraftKings posts passing props for {name}; using him as the starter "
                         f"(nflverse lists {games.at[row[0], f'{side}_qb_name']})")
            games.at[row[0], f'{side}_qb_id'] = gsis
            games.at[row[0], f'{side}_qb_name'] = name
    return games


def week_forecasts(frame, models, season, week):
    """{(gsis_id, game_id, market): (forecast, frame row)} for this week's upcoming rows in each market's population."""
    up = frame[~frame['played'] & (frame['season'] == season) & (frame['week'] == week)]
    out = {}
    for market in pl.MARKETS:
        rows = up[pm.population(up, pm.MARKET_GROUPS[market])]
        for (_, row), value in zip(rows.iterrows(), pm.forecast(models, rows, market)):
            out[(row['player_id'], row['game_id'], market)] = (float(value), row)
    return out


def score_week(lines, forecasts, dists, season):
    """This week's lines with forecasts, scored by the model (lines without a forecast are dropped)."""
    keep = [(g, gid, m) in forecasts for g, gid, m in zip(lines['gsis_id'], lines['game_id'], lines['market'])]
    rows = lines[keep].copy()
    if rows.empty:
        return rows
    rows['forecast'] = [forecasts[(g, gid, m)][0] for g, gid, m in zip(rows['gsis_id'], rows['game_id'], rows['market'])]
    rows['season'] = season
    return ps.score_lines(rows, {season: dists})


def closing_lookup(line_history, now=None):
    """{(game_id, gsis_id, market): closing line} for games that have kicked off.

    From ESPN (lines last moved before kickoff) and, failing that, this project's last line snapshot before kickoff.
    """
    now = now or pd.Timestamp.now(tz='UTC')
    closes = {}
    if os.path.exists(pl.SNAPSHOT_LOG):
        snaps = pd.read_csv(pl.SNAPSHOT_LOG, dtype={'espn_event': str, 'espn_athlete': str})
        if {'gsis_id', 'kickoff', 'logged_at'} <= set(snaps.columns):
            kick = pd.to_datetime(snaps['kickoff'], utc=True)
            logged = pd.to_datetime(snaps['logged_at']).dt.tz_localize('America/New_York', ambiguous='NaT',
                                                                         nonexistent='NaT').dt.tz_convert('UTC')
            snaps = snaps[(logged < kick) & (kick <= now)].assign(_logged=logged).sort_values('_logged')
            for r in snaps.itertuples():
                closes[(r.game_id, r.gsis_id, r.market)] = float(r.line)
    pre = line_history[line_history['pregame'].astype(str).str.lower() == 'true']
    for r in pre.itertuples():
        closes[(r.game_id, r.gsis_id, r.market)] = float(r.line)
    return closes


def grade_records(records, weeks, games, closes):
    """Fill result, W/L and closing line for logged picks whose games are final and whose stats are published."""
    final = set(games.loc[games['result'].notna(), 'game_id'])
    published = set(weeks['game_id'])
    stats = weeks.set_index(['game_id', 'player_id'])
    graded = 0
    for record in records:
        key = (record.get('game_id'), record.get('gsis_id'), record.get('market'))
        record['close'], record['clv'] = None, None
        if key in closes and record.get('line') is not None:
            record['close'] = closes[key]
            sign = 1 if record.get('side') == 'Under' else -1
            record['clv'] = round(sign * (float(record['line']) - closes[key]), 1)
        if record.get('outcome') not in (None, '') or key[0] not in final or key[0] not in published:
            continue
        if (key[0], key[1]) not in stats.index:
            record['outcome'], record['result'] = 'Void', None
        else:
            row = stats.loc[[(key[0], key[1])]]
            result = float(pm.outcome(row, record['market']).iloc[0])
            record['result'] = result
            record['outcome'] = ps.grade(record['side'], float(record['line']), result)
        graded += 1
    return graded


def make_pick(r, row, hist, kickoffs, season, week, note=None):
    """A pick record from a scored line `r` (itertuples row) and the player's feature row."""
    line = float(r.line)
    why = pm.reason(row, r.market)
    return {
        'season': season, 'week': week, 'kickoff': kickoffs[r.game_id][0], 'game': kickoffs[r.game_id][1],
        'player': r.player_name, 'team': row['team'], 'prop': pl.MARKETS[r.market][1],
        'pick': f"{r.side} {line:g}", 'line': line, 'open_line': r.open_line, 'forecast': round(r.forecast, 1),
        'chance': round(r.chance, 3), 'hist': hist, 'bet_at': ps.american_from_chance(hist, ps.MIN_EDGE),
        'why': (f"{note} " if note else '') + f"Forecast {r.forecast:.1f} (typical game {r.median:.1f}) vs line {line:g}"
               + (f"; {why}" if why else ''),
        'game_id': r.game_id, 'gsis_id': r.gsis_id, 'market': r.market, 'side': r.side,
    }


def qualifies(side, r, week):
    """Whether scored line `r` is a pick for `side`: overs only while the line is still at its known opener."""
    if side == 'Over':
        return (not pd.isna(r.open_line)
                and ps.is_over_pick(r.market, r.side, r.chance, week, line=r.line, open_line=r.open_line))
    return ps.is_pick(r.market, r.side, r.chance, week)


def final_reports(injuries, week_games, season, week):
    """The game ids of the week whose final injury report is in nflverse's data: a game status (Out, Doubtful or
    Questionable) for either team.  Game statuses appear only on a game's final report, which reaches nflverse by about
    9 AM Eastern the next day; before that the under model would run without the report the under rule was tested with."""
    if injuries is None or injuries.empty:
        return set()
    rows = injuries[(injuries['season'] == season) & (injuries['week'] == week)
                    & injuries['report_status'].isin(['Out', 'Doubtful', 'Questionable'])]
    teams = set(rows['team'].replace(TEAM_RENAMES))
    return {g.game_id for g in week_games.itertuples() if g.home_team in teams or g.away_team in teams}


def main(predict_year=None, predict_week=None, output_dir=None, log_dir=None, sides=DEFAULT_SIDES):
    """Find this week's prop picks for `sides` ('Over' and/or 'Under').

    predict_year / predict_week default to the current season's next unplayed week.
    """
    sides = [side for side in ('Over', 'Under') if side in sides]
    log_dir = log_dir or os.path.expanduser('~')
    log_filename = os.path.join(log_dir, f'nfl_props_{datetime.now():%Y%m%d_%H%M%S}.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()], force=True)
    logging.info(f"Logging to file: {log_filename}")
    run = RunLog(TOTAL_STEPS)
    season = predict_year or sheet.get_current_season_year()

    with run.step(1, "Loading nflverse data (player stats, schedule, injury reports, player ids)"):
        weeks, games, injuries = pm.load_inputs(season, refresh=True)
        players = nd.load_players(refresh=True)
        if not (games['season'] == season).any():
            raise ValueError(f"nflverse has no {season} schedule yet")
        week = predict_week or find_next_week_to_predict(games, season)
        week_games = games[(games['season'] == season) & (games['week'] == week)].copy()
        for column in ('home_team', 'away_team'):
            week_games[column] = week_games[column].replace(TEAM_RENAMES)
    logging.info(f"Prop picks for: Year {season}, Week {week} ({len(week_games)} games)")

    with run.step(2, "Fetching DraftKings prop lines from ESPN for the games not yet started"):
        lines, skipped = pl.fetch_week_lines(week_games)
        lines = pl.attach_players(lines, players) if len(lines) else lines.assign(gsis_id=None, player_name=None, position=None)
        logging.info(f"{len(lines)} prop lines for {lines['game_id'].nunique() if len(lines) else 0} games"
                     + (f"; not evaluated (already kicked off): {', '.join(skipped)}" if skipped else ''))

    with run.step(3, f"Building pre-game features and fitting the models on {season - pm.TRAIN_SEASONS}-{season - 1}"):
        upcoming = upcoming_rows(lines, week_games, weeks, players, season, week)
        games = market_starters(games, upcoming, lines, season, week)
        variants = {}
        for side in sides:
            early = RULES[side]['early']
            source = games
            if early:
                # opening game lines for every game played or posted so far (later weeks have none yet)
                posted = games[(games['season'] < season) | (games['week'] <= week)]
                source = pm.with_openers(games, pl.load_game_openers(posted, list(range(pl.FIRST_PROP_SEASON, season + 1))))
            frame = pm.build_frame(weeks, source, injuries, upcoming, early=early)
            variants[side] = {'frame': frame, 'models': pm.fit_models(frame, season)}
            logging.info(f"Fitted the {'early-week' if early else 'full'} model for the {side.lower()}s")

    with run.step(4, "Scoring past prop lines (the backtest record behind Hist. Win %) and the outcome distributions"):
        # every season with lines needs the forecasts of the DIST_SEASONS seasons before it
        first = min(season, pl.FIRST_PROP_SEASON) - pm.DIST_SEASONS
        line_history = pl.attach_players(pl.load_line_history(games, list(range(pl.FIRST_PROP_SEASON, season + 1))), players)
        # games each player's team played since his previous game, for the over rule's returner skip
        appearances = weeks[weeks['position'].isin(pm.OFFENSE_POSITIONS)][['player_id', 'game_id', 'team']]
        appearances = appearances.drop_duplicates(['player_id', 'game_id'])
        history_missed = appearances.assign(missed=pm.games_missed(weeks, games, appearances)).rename(
            columns={'player_id': 'gsis_id'})[['gsis_id', 'game_id', 'missed']]
        for side in sides:
            v, rule = variants[side], RULES[side]
            forecasts = pd.concat([pm.forecast_history(v['frame'], list(range(first, season)),
                                                       variant='early' if rule['early'] else ''),
                                   pm.season_forecasts(v['frame'], season, v['models'])], ignore_index=True)
            v['dists'] = pm.distributions(forecasts, season)
            scored_history = ps.score_history(line_history, forecasts, rule['graded_at'])
            history = rule['picks'](scored_history.merge(history_missed, on=['gsis_id', 'game_id'], how='left'))
            v['rates'], v['overall'] = ps.market_win_rates(history, markets=rule['markets'])
            logging.info(f"Backtest record of the {side.lower()} rule ({rule['label']}, 2024 on): " + ', '.join(
                f"{m} {w}-{l}" for m, (_, w, l) in v['rates'].items())
                + (f"; {v['overall']:.1%} overall" if v['overall'] else ''))
            if side == 'Over':
                v['late_rate'], late_n = ps.never_moved_rate(history)
                if v['late_rate'] is not None:
                    logging.info(f"Overs whose line never moved from its opener won {v['late_rate']:.1%} of {late_n}: "
                                 f"the rate a late over (within {ps.OVER_LATE_HOURS} hours of kickoff) is priced at")

    with run.step(5, "Forecasting this week's players and picking the lines"):
        kickoffs = {g.game_id: (kickoff_label(g.gameday, g.gametime), f"{g.away_team} at {g.home_team}")
                    for g in week_games.itertuples()}
        now = pd.Timestamp.now(tz='UTC')
        hours_left = {g.game_id: (pl.kickoff_utc(g.gameday, g.gametime) - now).total_seconds() / 3600
                      for g in week_games.itertuples()}
        week_missed = dict(zip(zip(upcoming['player_id'], upcoming['game_id']), pm.games_missed(weeks, games, upcoming))) \
            if len(upcoming) else {}
        waiting = [kickoffs[g][1] for g, hours in hours_left.items() if hours > ps.UNDER_WINDOW_HOURS]
        reported = final_reports(injuries, week_games, season, week)
        no_report = set()
        picks, returners = [], []
        for side in sides:
            v = variants[side]
            week_fc = week_forecasts(v['frame'], v['models'], season, week)
            scored = score_week(lines, week_fc, v['dists'], season) if len(lines) else lines
            side_picks = []
            for r in scored.itertuples() if len(scored) else []:
                if qualifies(side, r, week):
                    if side == 'Under' and hours_left[r.game_id] > ps.UNDER_WINDOW_HOURS:
                        continue
                    if side == 'Under' and r.game_id not in reported:
                        no_report.add(kickoffs[r.game_id][1])
                        continue
                    missed = week_missed.get((r.gsis_id, r.game_id))
                    if side == 'Over' and ps.returner_skip(missed, week):
                        returners.append((r, missed, week_fc[(r.gsis_id, r.game_id, r.market)][1]['team']))
                        continue
                    hist, note = v['rates'].get(r.market, (v['overall'], 0, 0))[0], None
                    if side == 'Over' and hours_left[r.game_id] < ps.OVER_LATE_HOURS and v.get('late_rate') is not None:
                        hist = v['late_rate']
                        note = (f"LATE: still at its opener {hours_left[r.game_id]:.0f}h before kickoff, so likely a line "
                                f"the market never moves (those won {hist:.1%}).")
                    side_picks.append(make_pick(r, week_fc[(r.gsis_id, r.game_id, r.market)][1], hist, kickoffs,
                                                season, week, note))
            side_picks.sort(key=lambda p: -p['chance'])
            if side == 'Over':
                logging.info(f"\n=== OVER PICKS, week {week}: overs the early-week model gives {ps.OVER_MIN_CHANCE:.0%}+ "
                             f"while the line is still at its opener ({len(side_picks)} of {len(scored)} lines) ===")
            else:
                logging.info(f"\n=== UNDER PICKS, week {week}: unders the full model gives {ps.MIN_CHANCE:.0%}+, in games "
                             f"within {ps.UNDER_WINDOW_HOURS} hours of kickoff ({len(side_picks)} of {len(scored)} lines) ===")
                if waiting:
                    logging.info(f"Unders are picked within {ps.UNDER_WINDOW_HOURS} hours of kickoff (after the final "
                                 f"injury report, near the closing line the rule was tested at): {len(waiting)} of "
                                 f"{len(hours_left)} games are further out; the daily run on their game day picks them up.")
                if no_report:
                    logging.info(f"No unders yet for {', '.join(sorted(no_report))}: its final injury report is not in "
                                 f"nflverse yet (it usually is by 9 AM Eastern the day after it is released).")
            for p in side_picks:
                bet_at = f"{p['bet_at']:+d} or better" if p['bet_at'] is not None else "a price you judge fair"
                hist = f"{p['hist']:.1%}" if p['hist'] is not None else "n/a"
                logging.info(f"{p['player']} ({p['team']}) {p['pick']} {p['prop']} - {p['game']}, {p['kickoff']}: "
                             f"model {p['chance']:.0%}, backtest {hist}, bet at {bet_at}")
                logging.info(f"    Why: {p['why']}")
            if side == 'Over' and len(scored):
                for r, missed, team in returners:
                    logging.info(f"Skipped (back from absence): {r.player_name} ({team}) Over {float(r.line):g} "
                                 f"{pl.MARKETS[r.market][1]} - {kickoffs[r.game_id][1]}: his team has played {missed:g} "
                                 f"games since his last, and forecasts for players back from {ps.RETURNER_MISSED}+ missed "
                                 f"games ran 20-30% high (README section 11.8)")
                gone = sum(1 for r in scored.itertuples() if r.side == 'Over' and r.market in ps.OVER_MARKETS
                           and r.chance >= ps.OVER_MIN_CHANCE and not qualifies(side, r, week))
                if gone:
                    logging.info(f"{gone} more overs reach {ps.OVER_MIN_CHANCE:.0%} but their line has already moved "
                                 f"from its opener, so the edge has been bet away; run earlier in the week to catch them.")
            picks += side_picks
        if len(lines):
            logging.info("DraftKings' prices are not in the feed: take a pick only at its Bet At price or better.")

    with run.step(6, "Grading earlier picks and saving a snapshot of this week's lines"):
        path = props_sheet.workbook_path(season, output_dir)
        records = props_sheet.read_log(path)
        # a later run on the same day replaces that day's earlier picks: the day's last run is the one that counts.  Kept:
        # picks from earlier days, graded ones, ones you typed a price or note on, and ones whose game has kicked off
        today = sheet.today_eastern()
        started = {g for g, hours in hours_left.items() if hours <= 0}
        before = len(records)
        records = [r for r in records if not (
            str(r.get('logged_at') or '').startswith(today) and r.get('outcome') in (None, '')
            and r.get('price') in (None, '') and r.get('notes') in (None, '') and r.get('game_id') not in started)]
        replaced = before - len(records)
        known = {props_sheet.pick_key(r) for r in records}
        stamp = datetime.now().isoformat(timespec='seconds')
        added = 0
        for p in picks:
            if props_sheet.pick_key(p) not in known:
                records.append({**p, 'logged_at': stamp})
                added += 1
        records.sort(key=lambda r: (int(r.get('season') or season), int(r.get('week') or 0)))
        graded = grade_records(records, weeks, games, closing_lookup(line_history))
        logging.info(f"{added} picks logged" + (f" ({replaced} logged earlier today replaced by this run's)" if replaced else '')
                     + f", {graded} earlier picks graded")
        try:
            logged = pl.append_prop_snapshot(lines)
            logging.info(f"Logged {logged} prop lines to {pl.SNAPSHOT_LOG}")
        except Exception as e:
            logging.warning(f"Could not log the prop line snapshot: {e}")

    with run.step(7, "Writing the props workbook"):
        try:
            notes = [] if len(lines) else ["No DraftKings prop lines were found for the games left this week."]
            if returners:
                notes.append("Skipped (back from absence, the returner skip): " + '; '.join(
                    f"{r.player_name} Over {float(r.line):g} {pl.MARKETS[r.market][1]} (his team has played {missed:g} "
                    f"games since his last)" for r, missed, _ in returners) + ".")
            under_empty = None
            if 'Under' in sides and (waiting or no_report):
                later = []
                if waiting:
                    later.append(f"unders for the {len(waiting)} later game(s) ({', '.join(waiting)}) come on the daily "
                                 f"run within {ps.UNDER_WINDOW_HOURS} hours of their kickoff")
                if no_report:
                    later.append(f"unders for {', '.join(sorted(no_report))} wait for the final injury report to reach "
                                 "nflverse (usually by 9 AM Eastern the next day)")
                text = '; '.join(later)
                if any(p['side'] == 'Under' for p in picks):
                    notes.append(text[0].upper() + text[1:] + '.')
                else:
                    under_empty = f"None yet: {text}."
            note = ' '.join(notes) or None
            rule_records = [(f"Record of the {side.lower()} rule ({RULES[side]['label']}, 2024 on), by market",
                             variants[side]['rates'], variants[side]['overall']) for side in sides]
            props_sheet.write_workbook(path, season, week, picks, records, rule_records, RULE_LINES, EVIDENCE,
                                       skipped=skipped, note=note, sides=sides, under_empty=under_empty)
            logging.info(f"Props workbook updated: {path}")
        except Exception as e:
            logging.error(f"Could not write the props workbook (close it in Excel and run again): {e}")
    logging.info(f"Run complete in {run.elapsed():.0f}s")
    return picks


if __name__ == "__main__":
    # Configuration
    PREDICT_YEAR = None  # None = current season; set a year to override
    PREDICT_WEEK = None  # None = first week with unplayed games; set a week number to override
    SIDES = ('Over', 'Under')    # ('Over',) for the overs only; the unders suit bets placed close to kickoff

    main(predict_year=PREDICT_YEAR, predict_week=PREDICT_WEEK, sides=SIDES)
