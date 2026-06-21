from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Manager(db.Model):
    __tablename__ = "manager"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)

    team_seasons = db.relationship("TeamSeason", back_populates="manager", lazy="dynamic")
    matchups_as_1 = db.relationship("Matchup", foreign_keys="Matchup.manager1_id", back_populates="manager1", lazy="dynamic")
    matchups_as_2 = db.relationship("Matchup", foreign_keys="Matchup.manager2_id", back_populates="manager2", lazy="dynamic")

    def __repr__(self):
        return f"<Manager {self.name}>"


class Season(db.Model):
    __tablename__ = "season"

    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, unique=True, nullable=False)
    champion_id = db.Column(db.Integer, db.ForeignKey("manager.id"), nullable=True)

    champion = db.relationship("Manager", foreign_keys=[champion_id])
    team_seasons = db.relationship("TeamSeason", back_populates="season", lazy="dynamic", cascade="all, delete-orphan")
    matchups = db.relationship("Matchup", back_populates="season", lazy="dynamic", cascade="all, delete-orphan")
    trades = db.relationship("Trade", back_populates="season", lazy="dynamic", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Season {self.year}>"


class TeamSeason(db.Model):
    __tablename__ = "team_season"
    __table_args__ = (db.UniqueConstraint("season_id", "manager_id", name="uq_team_season"),)

    id = db.Column(db.Integer, primary_key=True)
    season_id = db.Column(db.Integer, db.ForeignKey("season.id"), nullable=False)
    manager_id = db.Column(db.Integer, db.ForeignKey("manager.id"), nullable=False)
    team_name = db.Column(db.String(120), nullable=True)
    wins = db.Column(db.Integer, default=0)
    losses = db.Column(db.Integer, default=0)
    ties = db.Column(db.Integer, default=0)
    points_for = db.Column(db.Float, default=0.0)
    points_against = db.Column(db.Float, default=0.0)
    rank = db.Column(db.Integer, nullable=True)
    final_place = db.Column(db.Integer, nullable=True)
    made_playoffs = db.Column(db.Boolean, default=False)
    moves = db.Column(db.Integer, default=0)

    season = db.relationship("Season", back_populates="team_seasons")
    manager = db.relationship("Manager", back_populates="team_seasons")

    @property
    def games(self):
        return self.wins + self.losses + self.ties

    @property
    def win_pct(self):
        return round((self.wins + 0.5 * self.ties) / self.games, 3) if self.games else 0.0

    @property
    def ppg(self):
        return round(self.points_for / self.games, 1) if self.games else 0.0

    @property
    def papg(self):
        return round(self.points_against / self.games, 1) if self.games else 0.0

    @property
    def diff(self):
        return round(self.points_for - self.points_against, 1)

    @property
    def diff_pg(self):
        return round(self.ppg - self.papg, 1)

    def __repr__(self):
        return f"<TeamSeason {self.manager.name} {self.season.year}>"


class Matchup(db.Model):
    __tablename__ = "matchup"

    id = db.Column(db.Integer, primary_key=True)
    season_id = db.Column(db.Integer, db.ForeignKey("season.id"), nullable=False)
    week = db.Column(db.Integer, nullable=False)
    manager1_id = db.Column(db.Integer, db.ForeignKey("manager.id"), nullable=False)
    manager2_id = db.Column(db.Integer, db.ForeignKey("manager.id"), nullable=False)
    score1 = db.Column(db.Float, nullable=False)
    score2 = db.Column(db.Float, nullable=False)
    is_playoff = db.Column(db.Boolean, default=False)
    matchup_type = db.Column(db.String(20), default="regular")

    season = db.relationship("Season", back_populates="matchups")
    manager1 = db.relationship("Manager", foreign_keys=[manager1_id], back_populates="matchups_as_1")
    manager2 = db.relationship("Manager", foreign_keys=[manager2_id], back_populates="matchups_as_2")


class Trade(db.Model):
    __tablename__ = "trade"

    id = db.Column(db.Integer, primary_key=True)
    season_id = db.Column(db.Integer, db.ForeignKey("season.id"), nullable=False)
    week = db.Column(db.Integer, nullable=True)
    trade_date = db.Column(db.Date, nullable=True)
    notes = db.Column(db.Text, nullable=True)

    season = db.relationship("Season", back_populates="trades")
    assets = db.relationship("TradeAsset", back_populates="trade", cascade="all, delete-orphan")


class TradeAsset(db.Model):
    __tablename__ = "trade_asset"

    id = db.Column(db.Integer, primary_key=True)
    trade_id = db.Column(db.Integer, db.ForeignKey("trade.id"), nullable=False)
    from_manager_id = db.Column(db.Integer, db.ForeignKey("manager.id"), nullable=False)
    to_manager_id = db.Column(db.Integer, db.ForeignKey("manager.id"), nullable=False)
    asset_name = db.Column(db.String(120), nullable=False)
    asset_type = db.Column(db.String(20), default="player")

    trade = db.relationship("Trade", back_populates="assets")
    from_manager = db.relationship("Manager", foreign_keys=[from_manager_id])
    to_manager = db.relationship("Manager", foreign_keys=[to_manager_id])
