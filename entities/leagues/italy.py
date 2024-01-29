from database.entities.leagues.league import League


class SerieA(League):
    def __init__(self):
        super().__init__(
            country='Italy',
            name='Seria-A',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/I1.csv',
            fixtures_url='https://footystats.org/italy/serie-a/fixtures'
        )


class SerieB(League):
    def __init__(self):
        super().__init__(
            country='Italy',
            name='Seria-B',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/I2.csv',
            fixtures_url='https://footystats.org/italy/serie-b/fixtures'
        )
