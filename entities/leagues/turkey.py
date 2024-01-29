from database.entities.leagues.league import League


class SuperLig(League):
    def __init__(self):
        super().__init__(
            country='Turkey',
            name='Super-Lig',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/T1.csv',
            fixtures_url='https://footystats.org/turkey/super-lig/fixtures'
        )
