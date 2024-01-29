from database.entities.leagues.league import League


class IrelandPremierDivision(League):
    def __init__(self):
        super().__init__(
            country='Ireland',
            name='Premier-Division',
            year_start=2012,
            category='extra',
            data_url='https://www.football-data.co.uk/new/IRL.csv',
            fixtures_url='https://footystats.org/republic-of-ireland/premier-division/fixtures'
        )
