from datetime import date


class League:
    def __init__(
            self,
            country: str,
            league_name: str,
            base_url: str,
            year_start: int,
            league_type: str,
            fixtures_url: str
    ):
        self._country = country
        self._league_name = league_name
        self._base_url = base_url
        self._year_start = year_start
        self._league_type = league_type
        self._fixtures_url = fixtures_url

    @property
    def country(self) -> str:
        return self._country

    @property
    def league_name(self) -> str:
        return self._league_name

    @property
    def full_league_name(self) -> str:
        return self.country + ': ' + self.league_name

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def year_start(self) -> int:
        return self._year_start

    @property
    def league_type(self) -> str:
        return self._league_type

    @property
    def fixtures_url(self) -> str:
        return self._fixtures_url

    @property
    def url_list(self) -> list:
        return [
            self.base_url.format(str(year)[2:] + str(year + 1)[2:])
            for year in range(self.year_start, date.today().year + 1)
        ]
