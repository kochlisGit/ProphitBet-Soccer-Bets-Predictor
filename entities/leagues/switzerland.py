from database.entities.leagues.league import League


class SwitzerlandSuperLeague(League):
    def __init__(self):
        super().__init__(
            country='Sweden',
            name='Super-League',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/SWZ.csv',
            fixtures_url='https://footystats.org/switzerland/super-league/fixtures'
        )
