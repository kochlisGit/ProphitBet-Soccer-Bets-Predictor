from database.entities.leagues.league import League


class Allsvenskan(League):
    def __init__(self):
        super().__init__(
            country='Sweden',
            name='Allsvenskan',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/SWE.csv',
            fixtures_url='https://footystats.org/sweden/allsvenskan/fixtures'
        )
