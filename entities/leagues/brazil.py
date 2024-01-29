from database.entities.leagues.league import League


class BrazilSerieA(League):
    def __init__(self):
        super().__init__(
            country='Brazil',
            name='Serie-A',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/BRA.csv',
            fixtures_url='https://footystats.org/brazil/serie-a/fixtures'
        )
