from database.entities.leagues.league import League


class SuperLeague(League):
    def __init__(self):
        super().__init__(
            country='Greece',
            name='Super-League',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/G1.csv',
            fixtures_url='https://footystats.org/greece/super-league/fixtures'
        )
