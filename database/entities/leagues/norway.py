from database.entities.leagues.league import League


class Eliteserien(League):
    def __init__(self):
        super().__init__(
            country='Norway',
            name='Eliteserien',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/NOR.csv',
            fixtures_url='https://footystats.org/norway/eliteserien/fixtures'
        )
