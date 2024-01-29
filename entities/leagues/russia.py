from database.entities.leagues.league import League


class RussiaPremierLeague(League):
    def __init__(self):
        super().__init__(
            country='Russia',
            name='Premier-League',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/RUS.csv',
            fixtures_url='https://footystats.org/russia/russian-premier-league/fixtures'
        )
