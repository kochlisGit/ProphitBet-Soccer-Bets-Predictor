from database.entities.leagues.league import League


class PrimeraDivision(League):
    def __init__(self):
        super().__init__(
            country='Argentina',
            name='Primera-Division',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/ARG.csv',
            fixtures_url='https://footystats.org/argentina/primera-division/fixtures'
        )
