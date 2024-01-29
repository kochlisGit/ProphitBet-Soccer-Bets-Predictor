from database.entities.leagues.league import League


class Liga1(League):
    def __init__(self):
        super().__init__(
            country='Portugal',
            name='Liga-1',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/P1.csv',
            fixtures_url='https://footystats.org/portugal/liga-nos/fixtures'
        )
