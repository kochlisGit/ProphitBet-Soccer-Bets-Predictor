from database.entities.leagues.league import League


class JupilerLeague(League):
    def __init__(self):
        super().__init__(
            country='Belgium',
            name='Jupiler-League',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/B1.csv',
            fixtures_url='https://footystats.org/belgium/pro-league/fixtures'
        )
