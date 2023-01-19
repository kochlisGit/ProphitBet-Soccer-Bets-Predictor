class League:
    def __init__(
            self,
            country: str,
            name: str,
            url: str,
            year_start: int,
            league_type: str,
            fixtures_url: str
    ):
        self._country = country
        self._name = name
        self._url = url
        self._year_start = year_start
        self._league_type = league_type
        self._fixtures_url = fixtures_url

    @property
    def country(self) -> str:
        return self._country

    @property
    def name(self) -> str:
        return self._name

    @property
    def url(self) -> str:
        return self._url

    @property
    def year_start(self) -> int:
        return self._year_start

    @year_start.setter
    def year_start(self, year_start: int):
        self._year_start = year_start

    @property
    def league_type(self) -> str:
        return self._league_type

    @property
    def fixtures_url(self) -> str:
        return self._fixtures_url
