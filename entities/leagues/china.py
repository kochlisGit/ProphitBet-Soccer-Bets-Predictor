from database.entities.leagues.league import League


class ChinaSuperLeague(League):
    def __init__(self):
        super().__init__(
            country='China',
            name='Super-League',
            year_start=2014,
            category='extra',
            data_url='https://www.football-data.co.uk/new/CHN.csv',
            fixtures_url='https://footystats.org/china/chinese-super-league/fixtures'
        )
