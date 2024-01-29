from database.entities.leagues.league import League


class MLS(League):
    def __init__(self):
        super().__init__(
            country='USA',
            name='MLS',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/USA.csv',
            fixtures_url='https://footystats.org/usa/mls/fixtures'
        )
