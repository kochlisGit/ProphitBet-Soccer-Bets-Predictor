from database.entities.leagues.league import League


class Eredivisie(League):
    def __init__(self):
        super().__init__(
            country='Netherlands',
            name='Eredivisie',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/N1.csv',
            fixtures_url='https://footystats.org/netherlands/eredivisie/fixtures'
        )
