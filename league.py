from flask_wtf import FlaskForm
from wtforms import StringField, SelectMultipleField, widgets, SelectField, IntegerField
from wtforms.widgets import CheckboxInput
from wtforms.validators import InputRequired, NoneOf
from database.repositories.league import LeagueRepository


class CreateLeagueForm(FlaskForm):
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
                                     NoneOf(values=self._league_repository.get_all_saved_leagues(),
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

    def submit(self):
        league_name = self.league_name.data

        if not self._league_repository.league_exists(league_name=league_name):
            self._store_league()


    def _store_league(self):
        league_name = self.league_name.data
        last_n_matches = int(self.last_n_matches.data)
        goal_diff_margin = int(self.goal_diff_margin.data)

        selected_league_split = self.selected_league.data.replace(' ', '').split('-', maxsplit=1)
        selected_home_columns = self.home_columns.raw_data
        selected_away_columns = self.away_columns.raw_data
        matches_df, league = self._league_repository.create_league(
            league=self._all_leagues[(selected_league_split[0], selected_league_split[1])],
            last_n_matches=last_n_matches,
            goal_diff_margin=goal_diff_margin,
            statistic_columns=selected_home_columns + selected_away_columns,
            league_name=league_name
        )

        if matches_df is None:
            messagebox.showerror(
                'Internet Connection Error',
                'Application cannot connect to the internet. Check internet connection'
            )
        else:
            self._selected_league_and_matches_df = (league_name, league, matches_df)
