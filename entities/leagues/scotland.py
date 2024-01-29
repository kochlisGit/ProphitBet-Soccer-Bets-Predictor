from database.entities.leagues.league import League


class Premiership(League):
    def __init__(self):
        super().__init__(
            country='Scotland',
            name='Premiership',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/SC0.csv',
            fixtures_url='https://footystats.org/scotland/premiership/fixtures'
        )
