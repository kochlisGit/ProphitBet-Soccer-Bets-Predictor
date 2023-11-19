from pandas import DataFrame
from sqlalchemy import MetaData, text
from sqlalchemy.ext.declarative import declarative_base

from . import db
from .models import AvailableLeague, League


class DBWrapper:
    def get_league_matches(self, league_name: str):
        query = text(f"SELECT * FROM '{league_name}'")
        result = db.session.execute(query).fetchall()
        return DataFrame(list(result))

    def get_fixture_url_from_league_name(self, league_name: str):
        query = text(f"SELECT fixtures_url FROM available_league al JOIN league l on l.available_league_id=al.id WHERE l.name = '{league_name}'")
        result = db.session.execute(query).fetchall()
        return result[0][0]

    def league_exists(self, league_name: str):
        return (
            db.session.query(League.name).filter_by(name=league_name).first()
            is not None
        )

    def insert_league(self, league: League):
        db.session.add(league)
        db.session.commit()

    def delete_league(self, league_name: str):
        db.session.delete(League.query.filter_by(name=league_name).first())
        db.session.commit()

    def create_table_from_dataframe(self, df, table_name):
        df.to_sql(table_name, db.engine, index=False)

    def create_dataframe_from_table(self, table_name):
        query = text(f"SELECT * FROM '{table_name}'")
        result = db.session.execute(query).fetchall()
        return DataFrame(list(result))

    def drop_table(self, table_name: str):
        Base = declarative_base()
        metadata = MetaData()
        metadata.reflect(bind=db.engine)
        table = metadata.tables[table_name]
        if table is not None:
            Base.metadata.drop_all(db.engine, [table], checkfirst=True)
