from flask_wtf import FlaskForm
from wtforms import StringField, SelectMultipleField, SelectField, IntegerField, BooleanField
from wtforms.validators import InputRequired, NoneOf
from database.repositories.league import LeagueRepository
import pandas as pd
from website import db
from website.models import League
from database.network.netutils import check_internet_connection
from database.network.footballdata.extra import ExtraLeagueAPI
from database.network.footballdata.main import MainLeagueAPI
from preprocessing.statistics import StatisticsEngine

from sqlalchemy import MetaData, select, text
from sqlalchemy.ext.declarative import declarative_base


class LeagueForm(FlaskForm):
    def _get_all_saved_league_names(self):
        return [l.name for l in League.query.all()]

    def _league_exists(self):
        return db.session.query(League.name).filter_by(name=self.selected_league.data).first() is not None

    def _delete_league(self):
        league_name = self.selected_league.data
        db.session.delete(League.query.filter_by(name=league_name).first())
        db.session.commit()
        self._drop_league_table()
        self.selected_league.choices.remove(league_name)

    def _drop_league_table(self):
        table_name = self.selected_league.data
        Base = declarative_base()
        metadata = MetaData()
        metadata.reflect(bind=db.engine)
        table = metadata.tables[table_name]
        if table is not None:
            Base.metadata.drop_all(db.engine, [table], checkfirst=True)

    def _create_dataset(
            self,
            league: League,
            last_n_matches: int,
            goal_diff_margin: int,
            statistic_columns: list) -> pd.DataFrame or None:
        if check_internet_connection():
            matches_df = MainLeagueAPI().download(league=league)

            matches_df = StatisticsEngine(matches_df=matches_df,
                                          last_n_matches=last_n_matches,
                                          goal_diff_margin=goal_diff_margin
                                          ).compute_statistics(statistic_columns=statistic_columns)
            matches_df.to_sql(self.league_name.data, db.engine)
            return matches_df
        return None



class CreateLeagueForm(LeagueForm):
    selected_league = SelectField('Select League', validators=[InputRequired()])
    league_name = StringField('League Name')
    last_n_matches = IntegerField('Last N Matches Lookup', default=3, validators=[InputRequired()])
    goal_diff_margin = IntegerField('Goal Diff Margin', default=2, validators=[InputRequired()])
    home_columns = SelectMultipleField('Home Columns')
    away_columns = SelectMultipleField('Away Columns')

    def __init__(self, league_repository: LeagueRepository,  *args, **kwargs):
        super(CreateLeagueForm, self).__init__(*args, **kwargs)
        self._league_repository = league_repository
        self.league_name.validators=[InputRequired(),
                                     NoneOf(values=self._get_all_saved_league_names(),
                                            message="Name already exists")]
        self._all_leagues = league_repository.get_all_available_leagues()
        self.selected_league.choices = ["-".join(league) for league in self._all_leagues]
        self.selected_league.default = 10
        all_columns = league_repository.get_all_available_columns()
        self._home_columns = [col for col in all_columns if col[0] =='H']
        self._away_columns = [col for col in all_columns if col[0] == 'A']
        self.home_columns.choices = [(col, col) for col in self._home_columns]
        self.home_columns.data = [col for col in self._home_columns]
        self.away_columns.choices = [(col, col) for  col in self._away_columns]
        self.away_columns.data = [col for col in self._away_columns]

    def submit(self) -> (str, pd.DataFrame):
        if not self._league_exists():
            return self._store_league()
        return (None, None)

    def _store_league(self) -> (str, pd.DataFrame):
        league_name = self.league_name.data
        last_n_matches = int(self.last_n_matches.data)
        goal_diff_margin = int(self.goal_diff_margin.data)

        selected_league_split = self.selected_league.data.replace(' ', '').split('-', maxsplit=1)
        selected_home_columns = self.home_columns.raw_data
        selected_away_columns = self.away_columns.raw_data
        new_league = League(country=selected_league_split[0],
                            name=league_name,
                            last_n_matches=last_n_matches,
                            goal_diff_margin=goal_diff_margin,
                            statistic_columns= "::".join(selected_home_columns + selected_away_columns))
        db.session.add(new_league)
        db.session.commit()
        matches_df = self._create_dataset(
            league=self._all_leagues[(selected_league_split[0], selected_league_split[1])],
            last_n_matches=last_n_matches,
            goal_diff_margin=goal_diff_margin,
            statistic_columns=selected_home_columns + selected_away_columns,
        )

        return (league_name, matches_df)

    def _create_dataset(
            self,
            league: League,
            last_n_matches: int,
            goal_diff_margin: int,
            statistic_columns: list) -> pd.DataFrame or None:
        if check_internet_connection():
            matches_df = MainLeagueAPI().download(league=league)

            matches_df = StatisticsEngine(matches_df=matches_df,
                                          last_n_matches=last_n_matches,
                                          goal_diff_margin=goal_diff_margin
                                          ).compute_statistics(statistic_columns=statistic_columns)
            matches_df.to_sql(self.league_name.data, db.engine)
            return matches_df
        return None


class LoadLeagueForm(LeagueForm):
    selected_league = SelectField('Select League', validators=[InputRequired()])
    update_league = BooleanField('Update league')

    def __init__(self,  *args, **kwargs):
        super(LoadLeagueForm, self).__init__(*args, **kwargs)
        self._all_leagues = self._get_all_saved_league_names()
        self.selected_league.choices = self._all_leagues

    def submit(self):
        return self.selected_league.data, self._load_league()

    def _load_league(self) -> pd.DataFrame or None:
        if self._league_exists():
            query = text(f"SELECT * FROM '{self.selected_league.data}'")
            result = db.session.execute(query).fetchall()
            matches_df = pd.DataFrame(list(result))
            return matches_df
        else:
            return None


class DeleteLeagueForm(LeagueForm):
    selected_league = SelectField('Select League', validators=[InputRequired()])

    def __init__(self,  *args, **kwargs):
        super(DeleteLeagueForm, self).__init__(*args, **kwargs)

        self.selected_league.choices = self._get_all_saved_league_names()

    def submit(self):
        self._delete_league()