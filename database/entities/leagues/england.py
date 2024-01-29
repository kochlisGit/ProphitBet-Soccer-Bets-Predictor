from database.entities.leagues.league import League


class PremierLeague(League):
    def __init__(self):
        super().__init__(
            country='England',
            name='Premier-League',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/E0.csv',
            fixtures_url='https://footystats.org/england/premier-league/fixtures'
        )


class Championship(League):
    def __init__(self):
        super().__init__(
            country='England',
            name='Championship',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/E1.csv',
            fixtures_url='https://footystats.org/england/championship/fixtures'
        )


class League1(League):
    def __init__(self):
        super().__init__(
            country='England',
            name='League-1',
            year_start=2005,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/E2.csv',
            fixtures_url='https://footystats.org/england/efl-league-one/fixtures'
        )


class League2(League):
    def __init__(self):
        super().__init__(
            country='England',
            name='League-2',
            year_start=2004,
            category='main',
            data_url='https://www.football-data.co.uk/mmz4281/{}/E3.csv',
            fixtures_url='https://footystats.org/england/efl-league-two/fixtures'
        )
