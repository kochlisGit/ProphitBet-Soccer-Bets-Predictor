from flask_login import UserMixin
from sqlalchemy import event
from sqlalchemy.sql import func
from werkzeug.security import generate_password_hash

from . import db


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    league = db.relationship("League")


class mlModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    league_id = db.Column(db.Integer, db.ForeignKey("league.id"))


class League(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    last_n_matches = db.Column(db.Integer)
    goal_diff_margin = db.Column(db.Integer)
    statistic_columns = db.Column(db.String(1000))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    available_league_id = db.Column(db.Integer, db.ForeignKey("available_league.id"), nullable=False)
    ml_model = db.relationship("mlModel")


class AvailableLeague(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    country = db.Column(db.String(150))
    name = db.Column(db.String(150))
    league_type = db.Column(db.String(10))
    year_start = db.Column(db.Integer)
    url = db.Column(db.String(300))
    fixtures_url = db.Column(db.String(300))


@event.listens_for(User.__table__, "after_create")
def create_users(*args, **kwargs):
    db.session.add(
        User(
            first_name="maci",
            email="abc@lol.com",
            password=generate_password_hash("1234567"),
        )
    )
    db.session.commit()


@event.listens_for(AvailableLeague.__table__, "after_create")
def create_available_leagues(*args, **kwargs):
    data = [
        {
            "country": "Spain",
            "name": "La-Liga",
            "url": "https://www.football-data.co.uk/mmz4281/{}/SP1.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/spain/la-liga/fixtures",
        },
        {
            "country": "Spain",
            "name": "La-Liga2a",
            "url": "https://www.football-data.co.uk/mmz4281/{}/SP2.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/spain/segunda-division/fixtures",
        },
        {
            "country": "Belgium",
            "name": "Jupiler-League",
            "url": "https://www.football-data.co.uk/mmz4281/{}/B1.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/belgium/pro-league/fixtures",
        },
        {
            "country": "Brazil",
            "name": "Serie-A",
            "url": "https://www.football-data.co.uk/new/BRA.csv",
            "year_start": 2012,
            "league_type": "extra",
            "fixtures_url": "https://footystats.org/brazil/serie-a/fixtures",
        },
        {
            "country": "England",
            "name": "Premier-League",
            "url": "https://www.football-data.co.uk/mmz4281/{}/E0.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/england/premier-league/fixtures",
        },
        {
            "country": "France",
            "name": "Le-Championnat",
            "url": "https://www.football-data.co.uk/mmz4281/{}/F1.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/france/ligue-1/fixtures",
        },
        {
            "country": "Germany",
            "name": "Bundesliga-1",
            "url": "https://www.football-data.co.uk/mmz4281/{}/D1.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/germany/bundesliga/fixtures",
        },
        {
            "country": "Greece",
            "name": "Super-League",
            "url": "https://www.football-data.co.uk/mmz4281/{}/G1.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/greece/super-league/fixtures",
        },
        {
            "country": "Italy",
            "name": "Serie-A",
            "url": "https://www.football-data.co.uk/mmz4281/{}/I1.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/italy/serie-a/fixtures",
        },
        {
            "country": "Netherlands",
            "name": "Eredivisie",
            "url": "https://www.football-data.co.uk/mmz4281/{}/N1.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/netherlands/eredivisie/fixtures",
        },
        {
            "country": "Portugal",
            "name": "Liga-I",
            "url": "https://www.football-data.co.uk/mmz4281/{}/P1.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/portugal/liga-nos/fixtures",
        },
        {
            "country": "Scotland",
            "name": "Premiership",
            "url": "https://www.football-data.co.uk/mmz4281/{}/SC0.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/scotland/premiership/fixtures",
        },
        {
            "country": "Sweden",
            "name": "Allsvenskan",
            "url": "https://www.football-data.co.uk/new/SWE.csv",
            "year_start": 2012,
            "league_type": "extra",
            "fixtures_url": "https://footystats.org/sweden/allsvenskan/fixtures",
        },
        {
            "country": "Turkey",
            "name": "Futbol-Ligi-1",
            "url": "https://www.football-data.co.uk/mmz4281/{}/T1.csv",
            "year_start": 2012,
            "league_type": "main",
            "fixtures_url": "https://footystats.org/turkey/super-lig/fixtures",
        },
    ]

    for league_data in data:
        league = AvailableLeague(**league_data)
        db.session.add(league)

    db.session.commit()
