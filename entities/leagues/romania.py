from database.entities.leagues.league import League


class RomaniaLiga1(League):
    def __init__(self):
        super().__init__(
            country='Romania',
            name='Liga-1',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/ROU.csv',
            fixtures_url='https://footystats.org/romania/liga-i/fixtures'
        )
