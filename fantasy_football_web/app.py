from __future__ import annotations

import csv
import io
import os
from collections import defaultdict

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

from models import Manager, Matchup, Season, TeamSeason, Trade, TradeAsset, db
from seed_data import LEAGUE_NAME, MANAGERS, seed_database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "fantasy_football.db")


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FF_SECRET_KEY", "dev-secret-key")
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    with app.app_context():
        db.create_all()
        seed_database(db)

    @app.context_processor
    def inject_globals():
        return {"league_name": LEAGUE_NAME, "manager_names": MANAGERS}

    @app.route("/")
    def index():
        summaries = build_manager_summaries()
        seasons = Season.query.order_by(Season.year.desc()).all()
        champions = [{"year": season.year, "name": season.champion.name if season.champion else "TBD"} for season in seasons[:6]]
        overview = {
            "seasons": len(seasons),
            "managers": Manager.query.count(),
            "trades": Trade.query.count(),
            "matchups": Matchup.query.count(),
        }
        return render_template("index.html", summaries=summaries, champions=champions, overview=overview)

    @app.route("/history")
    def history():
        years = [season.year for season in Season.query.order_by(Season.year.desc()).all()]
        selected_year = request.args.get("year", type=int) or (years[0] if years else None)
        rows = []
        if selected_year is not None:
            rows = TeamSeason.query.join(Season).filter(Season.year == selected_year).order_by(TeamSeason.rank.asc(), TeamSeason.points_for.desc()).all()
        return render_template("history.html", years=years, selected_year=selected_year, rows=rows)

    @app.route("/records")
    def records():
        summaries = build_manager_summaries()
        best_ppg = TeamSeason.query.order_by(TeamSeason.points_for.desc()).limit(10).all()
        most_moves = TeamSeason.query.order_by(TeamSeason.moves.desc()).limit(10).all()
        return render_template("records.html", summaries=summaries, best_ppg=best_ppg, most_moves=most_moves)

    @app.route("/analytics")
    def analytics():
        return render_template("analytics.html")

    @app.route("/h2h")
    def h2h():
        manager1 = request.args.get("manager1", MANAGERS[0])
        manager2 = request.args.get("manager2", MANAGERS[1])
        summary, games = build_h2h(manager1, manager2)
        return render_template("h2h.html", manager1=manager1, manager2=manager2, summary=summary, games=games)

    @app.route("/trades")
    def trades():
        year = request.args.get("year", type=int)
        owner = request.args.get("owner", type=str)
        query = Trade.query.join(Season).order_by(Season.year.desc(), Trade.week.desc())
        if year:
            query = query.filter(Season.year == year)
        trade_rows = query.all()
        if owner:
            trade_rows = [trade for trade in trade_rows if any(asset.from_manager.name == owner or asset.to_manager.name == owner for asset in trade.assets)]
        years = [season.year for season in Season.query.order_by(Season.year.desc()).all()]
        return render_template("trades.html", trade_rows=trade_rows, years=years, selected_year=year, selected_owner=owner)

    @app.route("/import", methods=["GET", "POST"])
    def import_data():
        if request.method == "POST":
            dataset = request.form.get("dataset")
            csv_text = request.form.get("csv_text", "").strip()
            if not csv_text:
                flash("Paste CSV data before importing.", "error")
                return redirect(url_for("import_data"))
            try:
                if dataset == "team_summary":
                    import_team_summaries(csv_text)
                    flash("Team summary data imported.", "success")
                elif dataset == "trades":
                    import_trades(csv_text)
                    flash("Trade data imported.", "success")
                else:
                    flash("Unknown dataset type.", "error")
            except Exception as exc:  # pragma: no cover - manual import safety
                db.session.rollback()
                flash(f"Import failed: {exc}", "error")
            return redirect(url_for("import_data"))
        return render_template("import_data.html")

    @app.route("/api/overview")
    def api_overview():
        return jsonify(build_manager_summaries())

    @app.route("/api/win-pct")
    def api_win_pct():
        summaries = build_manager_summaries()
        return jsonify({
            "labels": [row["manager"] for row in summaries],
            "values": [round(row["win_pct"] * 100, 1) for row in summaries],
            "winning_seasons": [row["winning_seasons"] for row in summaries],
        })

    @app.route("/api/points-by-season")
    def api_points_by_season():
        grouped = defaultdict(list)
        seasons = Season.query.order_by(Season.year.asc()).all()
        for row in TeamSeason.query.join(Season).order_by(Season.year.asc()).all():
            grouped[row.manager.name].append({"year": row.season.year, "ppg": row.ppg, "wins": row.wins})
        return jsonify({"years": [season.year for season in seasons], "series": grouped})

    @app.route("/api/head-to-head-matrix")
    def api_head_to_head_matrix():
        matrix = []
        for owner1 in MANAGERS:
            row = []
            for owner2 in MANAGERS:
                if owner1 == owner2:
                    row.append(None)
                    continue
                summary, _ = build_h2h(owner1, owner2)
                row.append(summary["win_pct"] if summary else 0)
            matrix.append(row)
        return jsonify({"owners": MANAGERS, "matrix": matrix})

    @app.route("/api/h2h")
    def api_h2h():
        manager1 = request.args.get("manager1", MANAGERS[0])
        manager2 = request.args.get("manager2", MANAGERS[1])
        summary, games = build_h2h(manager1, manager2)
        return jsonify({"summary": summary, "games": games})

    return app


def build_manager_summaries() -> list[dict]:
    summaries = []
    for manager in Manager.query.order_by(Manager.name.asc()).all():
        seasons = manager.team_seasons.order_by(TeamSeason.season_id.asc()).all()
        wins = sum(row.wins for row in seasons)
        losses = sum(row.losses for row in seasons)
        ties = sum(row.ties for row in seasons)
        games = wins + losses + ties
        pf = sum(row.points_for for row in seasons)
        pa = sum(row.points_against for row in seasons)
        championships = sum(1 for row in seasons if row.final_place == 1)
        winning_seasons = sum(1 for row in seasons if row.wins > row.losses)
        summaries.append({
            "manager": manager.name,
            "seasons": len(seasons),
            "wins": wins,
            "losses": losses,
            "win_pct": round((wins + 0.5 * ties) / games, 3) if games else 0,
            "points_for": round(pf, 1),
            "points_against": round(pa, 1),
            "ppg": round(pf / games, 1) if games else 0,
            "papg": round(pa / games, 1) if games else 0,
            "championships": championships,
            "winning_seasons": winning_seasons,
            "median_rank": median_rank(seasons),
            "avg_moves": round(sum(row.moves for row in seasons) / len(seasons), 1) if seasons else 0,
        })
    return sorted(summaries, key=lambda row: (-row["win_pct"], -row["wins"], row["manager"]))


def median_rank(seasons: list[TeamSeason]) -> float:
    ranks = sorted(row.rank for row in seasons if row.rank is not None)
    if not ranks:
        return 0
    mid = len(ranks) // 2
    if len(ranks) % 2:
        return float(ranks[mid])
    return round((ranks[mid - 1] + ranks[mid]) / 2, 1)


def build_h2h(manager1: str, manager2: str):
    if manager1 == manager2:
        return None, []
    games = Matchup.query.join(Season).filter(
        ((Matchup.manager1.has(name=manager1)) & (Matchup.manager2.has(name=manager2)))
        | ((Matchup.manager1.has(name=manager2)) & (Matchup.manager2.has(name=manager1)))
    ).order_by(Season.year.desc(), Matchup.week.desc()).all()
    results = []
    wins1 = wins2 = 0
    points1 = points2 = 0.0
    for game in games:
        if game.manager1.name == manager1:
            score1, score2 = game.score1, game.score2
            opp_name = game.manager2.name
        else:
            score1, score2 = game.score2, game.score1
            opp_name = game.manager1.name
        winner = manager1 if score1 > score2 else opp_name
        wins1 += 1 if winner == manager1 else 0
        wins2 += 1 if winner != manager1 else 0
        points1 += score1
        points2 += score2
        results.append({
            "year": game.season.year,
            "week": game.week,
            "manager1_score": score1,
            "manager2_score": score2,
            "winner": winner,
            "is_playoff": game.is_playoff,
        })
    summary = {
        "manager1": manager1,
        "manager2": manager2,
        "games": len(results),
        "manager1_wins": wins1,
        "manager2_wins": wins2,
        "win_pct": round(wins1 / len(results), 3) if results else 0,
        "avg_score_for": round(points1 / len(results), 1) if results else 0,
        "avg_score_against": round(points2 / len(results), 1) if results else 0,
    }
    return summary, results


def parse_csv_text(csv_text: str):
    return csv.DictReader(io.StringIO(csv_text))


def get_or_create_manager(owner: str) -> Manager:
    manager = Manager.query.filter_by(name=owner).first()
    if manager is None:
        manager = Manager(name=owner)
        db.session.add(manager)
        db.session.flush()
    return manager


def import_team_summaries(csv_text: str) -> None:
    for row in parse_csv_text(csv_text):
        year = int(row["year"])
        owner = row["owner"].strip()
        season = Season.query.filter_by(year=year).first()
        if season is None:
            season = Season(year=year)
            db.session.add(season)
            db.session.flush()
        manager = get_or_create_manager(owner)
        existing = TeamSeason.query.filter_by(season_id=season.id, manager_id=manager.id).first()
        payload = {
            "team_name": row.get("team") or row.get("team_name") or owner,
            "wins": int(row.get("wins", 0)),
            "losses": int(row.get("losses", 0)),
            "ties": int(row.get("ties", 0)),
            "points_for": float(row.get("pf", row.get("points_for", 0))),
            "points_against": float(row.get("pa", row.get("points_against", 0))),
            "rank": int(row.get("rank", 0)) if row.get("rank") else None,
            "final_place": int(row.get("final_place", row.get("rank", 0))) if (row.get("final_place") or row.get("rank")) else None,
            "made_playoffs": str(row.get("made_playoffs", "")).lower() in {"1", "true", "yes", "y"},
            "moves": int(row.get("moves", 0)),
        }
        if existing:
            for key, value in payload.items():
                setattr(existing, key, value)
        else:
            db.session.add(TeamSeason(season=season, manager=manager, **payload))
    db.session.commit()


def import_trades(csv_text: str) -> None:
    trades_by_key = {}
    for row in parse_csv_text(csv_text):
        year = int(row["year"])
        season = Season.query.filter_by(year=year).first()
        if season is None:
            season = Season(year=year)
            db.session.add(season)
            db.session.flush()
        week = int(row.get("week", 0)) if row.get("week") else None
        key = (year, week, row.get("notes", "").strip())
        trade = trades_by_key.get(key)
        if trade is None:
            trade = Trade(season=season, week=week, notes=row.get("notes", "").strip() or None)
            db.session.add(trade)
            db.session.flush()
            trades_by_key[key] = trade
        from_manager = get_or_create_manager(row["from_owner"].strip())
        to_manager = get_or_create_manager(row["to_owner"].strip())
        db.session.add(TradeAsset(
            trade=trade,
            from_manager=from_manager,
            to_manager=to_manager,
            asset_name=row["asset_name"].strip(),
            asset_type=(row.get("asset_type") or "player").strip(),
        ))
    db.session.commit()


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
