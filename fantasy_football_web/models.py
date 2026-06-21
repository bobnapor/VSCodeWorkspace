from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Manager(db.Model):
    __tablename__ = 'manager'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)

    team_seasons = db.relationship('TeamSeason', back_populates='manager', lazy='dynamic')
    matchups_as_1 = db.relationship('Matchup', foreign_keys='Matchup.manager1_id', back_populates='manager1', lazy='dynamic')
    matchups_as_2 = db.relationship('Matchup', foreign_keys='Matchup.manager2_id', back_populates='manager2', lazy='dynamic')

    def __repr__(self):
        return f'<Manager {self.name}>'


class Season(db.Model):
    __tablename__ = 'season'
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, unique=True, nullable=False)
    champion_id = db.Column(db.Integer, db.ForeignKey('manager.id'), nullable=True)

    champion = db.relationship('Manager', foreign_keys=[champion_id])
    team_seasons = db.relationship('TeamSeason', back_populates='season', lazy='dynamic')
    matchups = db.relationship('Matchup', back_populates='season', lazy='dynamic')
    trades = db.relationship('Trade', back_populates='season', lazy='dynamic')

    def __repr__(self):
        return f'<Season {self.year}>'


class TeamSeason(db.Model):
    """One row per manager per season."""
    __tablename__ = 'team_season'
    id = db.Column(db.Integer, primary_key=True)
    season_id = db.Column(db.Integer, db.ForeignKey('season.id'), nullable=False)
    manager_id = db.Column(db.Integer, db.ForeignKey('manager.id'), nullable=False)
    team_name = db.Column(db.String(120), nullable=True)
    wins = db.Column(db.Integer, default=0)
    losses = db.Column(db.Integer, default=0)
    ties = db.Column(db.Integer, default=0)
    points_for = db.Column(db.Float, default=0.0)
    points_against = db.Column(db.Float, default=0.0)
    rank = db.Column(db.Integer, nullable=True)        # final regular-season rank
    final_place = db.Column(db.Integer, nullable=True)  # overall finish (1 = champion)
    made_playoffs = db.Column(db.Boolean, default=False)
    moves = db.Column(db.Integer, default=0)

    season = db.relationship('Season', back_populates='team_seasons')
    manager = db.relationship('Manager', back_populates='team_seasons')

    @property
    def win_pct(self):
        total = self.wins + self.losses + self.ties
        return round(self.wins / total, 3) if total > 0 else 0.0

    @property
    def ppg(self):
        total = self.wins + self.losses + self.ties
        return round(self.points_for / total, 1) if total > 0 else 0.0

    @property
    def papg(self):
        total = self.wins + self.losses + self.ties
        return round(self.points_against / total, 1) if total > 0 else 0.0

    @property
    def diff_pg(self):
        return round(self.ppg - self.papg, 1)

    def __repr__(self):
        return f'<TeamSeason {self.manager.name} {self.season.year}>'


class Matchup(db.Model):
    """One row per weekly matchup."""
    __tablename__ = 'matchup'
    id = db.Column(db.Integer, primary_key=True)
    season_id = db.Column(db.Integer, db.ForeignKey('season.id'), nullable=False)
    week = db.Column(db.Integer, nullable=False)
    manager1_id = db.Column(db.Integer, db.ForeignKey('manager.id'), nullable=False)
    manager2_id = db.Column(db.Integer, db.ForeignKey('manager.id'), nullable=False)
    score1 = db.Column(db.Float, nullable=False)
    score2 = db.Column(db.Float, nullable=False)
    is_playoff = db.Column(db.Boolean, default=False)
    matchup_type = db.Column(db.String(20), default='regular')  # regular / semifinal / championship / consolation

    season = db.relationship('Season', back_populates='matchups')
    manager1 = db.relationship('Manager', foreign_keys=[manager1_id], back_populates='matchups_as_1')
    manager2 = db.relationship('Manager', foreign_keys=[manager2_id], back_populates='matchups_as_2')

    def __repr__(self):
        return f'<Matchup {self.season.year} W{self.week}>'


class Trade(db.Model):
    __tablename__ = 'trade'
    id = db.Column(db.Integer, primary_key=True)
    season_id = db.Column(db.Integer, db.ForeignKey('season.id'), nullable=False)
    week = db.Column(db.Integer, nullable=True)
    trade_date = db.Column(db.Date, nullable=True)
    notes = db.Column(db.Text, nullable=True)

    season = db.relationship('Season', back_populates='trades')
    assets = db.relationship('TradeAsset', back_populates='trade', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Trade {self.id} Season {self.season.year}>'


class TradeAsset(db.Model):
    __tablename__ = 'trade_asset'
    id = db.Column(db.Integer, primary_key=True)
    trade_id = db.Column(db.Integer, db.ForeignKey('trade.id'), nullable=False)
    from_manager_id = db.Column(db.Integer, db.ForeignKey('manager.id'), nullable=False)
    to_manager_id = db.Column(db.Integer, db.ForeignKey('manager.id'), nullable=False)
    asset_name = db.Column(db.String(120), nullable=False)
    asset_type = db.Column(db.String(20), default='player')  # player / pick / other

    trade = db.relationship('Trade', back_populates='assets')
    from_manager = db.relationship('Manager', foreign_keys=[from_manager_id])
    to_manager = db.relationship('Manager', foreign_keys=[to_manager_id])
