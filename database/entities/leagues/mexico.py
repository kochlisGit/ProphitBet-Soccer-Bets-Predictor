from database.entities.leagues.league import League


class LigaMX(League):
    def __init__(self):
        super().__init__(
            country='Mexico',
            name='Liga-MX',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/MEX.csv',
            fixtures_url='https://footystats.org/mexico/liga-mx/fixtures'
        )
