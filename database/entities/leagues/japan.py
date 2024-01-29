from database.entities.leagues.league import League


class J1(League):
    def __init__(self):
        super().__init__(
            country='Japan',
            name='J-1',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/JPN.csv',
            fixtures_url='https://footystats.org/japan/j1-league/fixtures'
        )
