from __future__ import annotations

from datetime import date
from random import Random

from models import Manager, Matchup, Season, TeamSeason, Trade, TradeAsset

LEAGUE_NAME = "Skimbleshanks"
MANAGERS = ["Alec", "Ben", "Bobby", "Dan", "Fred", "James", "Jer", "Myles", "Nick", "Sean"]

YEAR_WINS = {
    2009: {"Alec": 6, "Ben": 8, "Bobby": 6, "Dan": 6, "Fred": 4, "James": 6, "Jer": 7, "Myles": 5, "Nick": 0, "Sean": 6},
    2010: {"Alec": 7, "Ben": 4, "Bobby": 3, "Dan": 5, "Fred": 5, "James": 7, "Jer": 5, "Myles": 2, "Nick": 0, "Sean": 8},
    2011: {"Alec": 5, "Ben": 4, "Bobby": 6, "Dan": 8, "Fred": 4, "James": 6, "Jer": 6, "Myles": 5, "Nick": 0, "Sean": 4},
    2012: {"Alec": 5, "Ben": 6, "Bobby": 7, "Dan": 10, "Fred": 4, "James": 9, "Jer": 6, "Myles": 7, "Nick": 6, "Sean": 5},
    2013: {"Alec": 8, "Ben": 7, "Bobby": 7, "Dan": 7, "Fred": 7, "James": 7, "Jer": 4, "Myles": 8, "Nick": 3, "Sean": 7},
    2014: {"Alec": 9, "Ben": 7, "Bobby": 12, "Dan": 2, "Fred": 7, "James": 8, "Jer": 4, "Myles": 3, "Nick": 5, "Sean": 8},
    2015: {"Alec": 6, "Ben": 7, "Bobby": 10, "Dan": 8, "Fred": 6, "James": 5, "Jer": 5, "Myles": 6, "Nick": 6, "Sean": 6},
    2016: {"Alec": 6, "Ben": 10, "Bobby": 3, "Dan": 3, "Fred": 8, "James": 5, "Jer": 8, "Myles": 3, "Nick": 10, "Sean": 9},
    2017: {"Alec": 7, "Ben": 5, "Bobby": 10, "Dan": 7, "Fred": 9, "James": 7, "Jer": 6, "Myles": 2, "Nick": 5, "Sean": 7},
    2018: {"Alec": 5, "Ben": 7, "Bobby": 8, "Dan": 7, "Fred": 6, "James": 6, "Jer": 5, "Myles": 7, "Nick": 5, "Sean": 9},
    2019: {"Alec": 6, "Ben": 7, "Bobby": 5, "Dan": 7, "Fred": 4, "James": 12, "Jer": 8, "Myles": 6, "Nick": 3, "Sean": 7},
    2020: {"Alec": 9, "Ben": 8, "Bobby": 5, "Dan": 7, "Fred": 6, "James": 9, "Jer": 2, "Myles": 4, "Nick": 7, "Sean": 8},
    2021: {"Alec": 5, "Ben": 8, "Bobby": 4, "Dan": 7, "Fred": 9, "James": 11, "Jer": 8, "Myles": 9, "Nick": 7, "Sean": 2},
    2022: {"Alec": 8, "Ben": 10, "Bobby": 10, "Dan": 2, "Fred": 6, "James": 2, "Jer": 9, "Myles": 9, "Nick": 5, "Sean": 9},
    2023: {"Alec": 5, "Ben": 8, "Bobby": 7, "Dan": 11, "Fred": 7, "James": 10, "Jer": 2, "Myles": 8, "Nick": 4, "Sean": 8},
}

VISIBLE_TEAM_NAMES = {
    (2020, "Alec"): "Thurman Merman",
    (2020, "Ben"): "Thread Level Midnight",
    (2020, "Bobby"): "The Algorithm",
    (2020, "Dan"): "Retreat or Suffer Defeat",
    (2020, "Fred"): "Kenny Kawaguchi",
    (2020, "James"): "Dalvin And The Chipmunks",
    (2020, "Jer"): "Julio Get The Stretch",
    (2020, "Myles"): "Fantasy Team Go Brrr",
    (2020, "Nick"): "Sofa King Pure",
    (2020, "Sean"): "Sunday Red",
    (2021, "Alec"): "Thurman Merman",
    (2021, "Ben"): "Threat Level Midnight",
    (2021, "Bobby"): "Toney the Tiger",
    (2021, "Dan"): "Colors that end in \"urple\"",
    (2021, "Fred"): "Jewish McCaffrey",
    (2021, "James"): "The Wallerboy",
    (2021, "Jer"): "HODL Beckham Jr",
    (2021, "Myles"): "Just Joshing",
    (2021, "Nick"): "Sofa King Pure",
    (2021, "Sean"): "Sunday Red",
    (2022, "Alec"): "Thurman Merman",
    (2022, "Ben"): "Threat Level Midnight",
    (2022, "Bobby"): "Man Box",
    (2022, "Dan"): "The Rebuild",
    (2022, "Fred"): "Oy Vey, JJ!",
    (2022, "James"): "Bust It",
    (2022, "Jer"): "Cali Fournetnication",
    (2022, "Myles"): "Sunday Red",
    (2022, "Nick"): "Flush Puppies",
    (2022, "Sean"): "Just Joshing",
    (2023, "Alec"): "Thurman Merman",
    (2023, "Ben"): "Threat Level Midnight",
    (2023, "Bobby"): "Techno Santa",
    (2023, "Dan"): "Zay my name",
    (2023, "Fred"): "Oy Vey, JJ!",
    (2023, "James"): "Titanic After Dark",
    (2023, "Jer"): "McPhearless (Tyreek's Version)",
    (2023, "Myles"): "Brown and Brown",
    (2023, "Nick"): "Flush Puppies",
    (2023, "Sean"): "Orchids of Asia",
}

CHAMPIONS = {
    2009: "Ben", 2010: "Sean", 2011: "Dan", 2012: "Dan", 2013: "Alec",
    2014: "Bobby", 2015: "Bobby", 2016: "Ben", 2017: "Fred", 2018: "Sean",
    2019: "James", 2020: "Alec", 2021: "Myles", 2022: "Bobby", 2023: "Fred",
}

TEAM_PREFIXES = ["Gridiron", "Sunday", "Velvet", "Moonshot", "Chaos", "Turbo", "Iron", "Frozen", "Golden", "Neon"]
TEAM_SUFFIXES = ["Bandits", "Wolves", "Titans", "Outlaws", "Monarchs", "Pirates", "Nomads", "Signal Callers", "Renegades", "Marauders"]
PLAYER_POOL = [
    "Christian McCaffrey", "Tyreek Hill", "Stefon Diggs", "Josh Allen", "Saquon Barkley", "Amon-Ra St. Brown",
    "CeeDee Lamb", "Breece Hall", "Jahmyr Gibbs", "A.J. Brown", "Mark Andrews", "Travis Kelce",
    "Jaylen Waddle", "Patrick Mahomes", "Puka Nacua", "Deebo Samuel", "Jonathan Taylor", "James Cook",
]


def season_games(year: int) -> int:
    if year >= 2021:
        return 14
    if year >= 2012:
        return 13
    return 10


def generated_team_name(year: int, manager: str) -> str:
    prefix = TEAM_PREFIXES[(year + len(manager)) % len(TEAM_PREFIXES)]
    suffix = TEAM_SUFFIXES[(year * 3 + len(manager)) % len(TEAM_SUFFIXES)]
    return f"{prefix} {suffix}"


def rank_map_for_year(year: int) -> dict[str, int]:
    wins = YEAR_WINS[year]
    ordered = sorted(MANAGERS, key=lambda name: (-wins[name], name))
    champion = CHAMPIONS[year]
    if champion in ordered:
        ordered.remove(champion)
        ordered.insert(0, champion)
    return {name: index + 1 for index, name in enumerate(ordered)}


def build_round_robin(owners: list[str]) -> list[list[tuple[str, str]]]:
    rotation = owners[:] 
    rounds = []
    for _ in range(len(rotation) - 1):
        pairs = []
        for i in range(len(rotation) // 2):
            pairs.append((rotation[i], rotation[-(i + 1)]))
        rounds.append(pairs)
        rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]
    return rounds


def create_matchups(season: Season, managers: dict[str, Manager], team_rows: dict[str, TeamSeason]) -> list[Matchup]:
    rounds = build_round_robin(MANAGERS)
    weeks_needed = season_games(season.year)
    rnd = Random(season.year)
    matchups = []
    for week in range(1, weeks_needed + 1):
        pairings = rounds[(week - 1) % len(rounds)]
        for owner1, owner2 in pairings:
            ts1 = team_rows[owner1]
            ts2 = team_rows[owner2]
            ppg1 = ts1.ppg or 115
            ppg2 = ts2.ppg or 115
            score1 = round(max(72.0, rnd.gauss(ppg1, 14)), 1)
            score2 = round(max(72.0, rnd.gauss(ppg2, 14)), 1)
            if score1 == score2:
                score2 += 0.1
            matchups.append(Matchup(
                season=season,
                week=week,
                manager1=managers[owner1],
                manager2=managers[owner2],
                score1=score1,
                score2=score2,
                is_playoff=week > weeks_needed - 2,
                matchup_type="playoff" if week > weeks_needed - 2 else "regular",
            ))
    return matchups


def create_trades(season: Season, managers: dict[str, Manager]) -> list[Trade]:
    rnd = Random(season.year * 17)
    trade_count = 2 + (season.year % 3)
    trades = []
    owner_names = list(managers.keys())
    for idx in range(trade_count):
        from_owner, to_owner = rnd.sample(owner_names, 2)
        trade = Trade(
            season=season,
            week=2 + idx * 3,
            trade_date=date(season.year, min(12, 9 + idx), 5 + idx),
            notes=f"Need-based swap between {from_owner} and {to_owner}.",
        )
        asset_out = PLAYER_POOL[(season.year + idx) % len(PLAYER_POOL)]
        asset_in = PLAYER_POOL[(season.year + idx + 5) % len(PLAYER_POOL)]
        trade.assets.append(TradeAsset(
            from_manager=managers[from_owner],
            to_manager=managers[to_owner],
            asset_name=asset_out,
            asset_type="player",
        ))
        trade.assets.append(TradeAsset(
            from_manager=managers[to_owner],
            to_manager=managers[from_owner],
            asset_name=asset_in,
            asset_type="player",
        ))
        if idx % 2 == 0:
            trade.assets.append(TradeAsset(
                from_manager=managers[from_owner],
                to_manager=managers[to_owner],
                asset_name=f"{season.year + 1} 3rd-round pick",
                asset_type="pick",
            ))
        trades.append(trade)
    return trades


def seed_database(db) -> None:
    if Manager.query.count() > 0:
        return

    manager_rows = {name: Manager(name=name) for name in MANAGERS}
    db.session.add_all(manager_rows.values())
    db.session.flush()

    for year in sorted(YEAR_WINS):
        season = Season(year=year, champion=manager_rows[CHAMPIONS[year]])
        db.session.add(season)
        db.session.flush()

        year_ranks = rank_map_for_year(year)
        games = season_games(year)
        team_rows = {}

        for owner in MANAGERS:
            wins = YEAR_WINS[year][owner]
            losses = max(games - wins, 0)
            rank = year_ranks[owner]
            rnd = Random(year * 100 + rank * 13 + len(owner))
            ppg = round(104 + wins * 1.7 + (11 - rank) * 1.35 + rnd.uniform(-4, 6), 1)
            papg = round(104 + losses * 1.4 + rank * 0.9 + rnd.uniform(-6, 5), 1)
            pf = round(ppg * games, 1)
            pa = round(papg * games, 1)
            row = TeamSeason(
                season=season,
                manager=manager_rows[owner],
                team_name=VISIBLE_TEAM_NAMES.get((year, owner), generated_team_name(year, owner)),
                wins=wins,
                losses=losses,
                ties=0,
                points_for=pf,
                points_against=pa,
                rank=rank,
                final_place=1 if owner == CHAMPIONS[year] else rank,
                made_playoffs=rank <= 6,
                moves=5 + (year + rank * 3 + len(owner)) % 31,
            )
            db.session.add(row)
            team_rows[owner] = row

        db.session.add_all(create_matchups(season, manager_rows, team_rows))
        db.session.add_all(create_trades(season, manager_rows))

    db.session.commit()
