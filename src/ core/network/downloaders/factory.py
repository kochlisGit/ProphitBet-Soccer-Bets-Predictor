from src.core.network.downloaders.downloader import LeagueDownloader
from src.core.network.downloaders.explorer import BetExplorerLeagueDownloader
from src.core.network.downloaders.extra import ExtraLeagueDownloader
from src.core.network.downloaders.main import MainLeagueDownloader


class DownloaderFactory:
    """ Factory class that returns the correct downloader based on league's category. """

    def __init__(self):
        self._downloaders = {
            'main': MainLeagueDownloader,
            'extra': ExtraLeagueDownloader,
            'explorer': BetExplorerLeagueDownloader
        }

    def get_downloader(self, category: str) -> LeagueDownloader:
        """ Returns the correct downloader based on league's category.

            Args
            ----
            :param category: (str) League's category. Currently, it supports "main", "extra" or "explorer".
        """

        downloader_instance = self._downloaders[category]
        return downloader_instance()
