from database.entities.leagues.league import League


class SuperLiga(League):
    def __init__(self):
        super().__init__(
            country='Denmark',
            name='Super-Liga',
            year_start=2013,
            category='extra',
            data_url='https://www.football-data.co.uk/new/DNK.csv',
            fixtures_url='https://footystats.org/denmark/1st-division/fixtures'
        )
