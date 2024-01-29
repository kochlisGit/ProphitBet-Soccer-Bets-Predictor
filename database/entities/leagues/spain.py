from database.entities.leagues.league import League


class LaLiga(League):
    def __init__(self):
        super().__init__(
            country='Spain',
            name='La-Liga',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/SP1.csv',
            fixtures_url='https://footystats.org/spain/la-liga/fixtures'
        )


class SegundaDivision(League):
    def __init__(self):
        super().__init__(
            country='Spain',
            name='Segunda-Division',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/SP2.csv',
            fixtures_url='https://footystats.org/spain/segunda-division/fixtures'
        )
