from database.entities.leagues.league import League


class Ekstraklasa(League):
    def __init__(self):
        super().__init__(
            country='Poland',
            name='Ekstraklasa',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/POL.csv',
            fixtures_url='https://footystats.org/poland/ekstraklasa/fixtures'
        )
