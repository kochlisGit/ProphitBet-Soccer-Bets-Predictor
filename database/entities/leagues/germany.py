from database.entities.leagues.league import League


class Bundesliga1(League):
    def __init__(self):
        super().__init__(
            country='Germany',
            name='Bundesliga-1',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/D1.csv',
            fixtures_url='https://footystats.org/germany/bundesliga/fixtures'
        )


class Bundesliga2(League):
    def __init__(self):
        super().__init__(
            country='Germany',
            name='Bundesliga-2',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/D2.csv',
            fixtures_url='https://footystats.org/germany/2-bundesliga/fixtures'
        )
