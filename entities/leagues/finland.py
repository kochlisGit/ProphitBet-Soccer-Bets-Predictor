from database.entities.leagues.league import League


class VeikkausLiiga(League):
    def __init__(self):
        super().__init__(
            country='Finland',
            name='Veikkausliiga',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/FIN.csv',
            fixtures_url='https://footystats.org/finland/veikkausliiga/fixtures'
        )
