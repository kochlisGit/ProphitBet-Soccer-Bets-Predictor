import warnings
import config
from database.repositories.league import LeagueRepository
from database.repositories.model import ModelRepository
from gui.main import MainApplicationWindow


def main():
    league_repository = LeagueRepository(
        leagues_directory=config.leagues_directory,
        leagues_index_filepath=config.leagues_index_filepath,
        all_leagues_dict=config.all_leagues_dict
    )
    model_repository = ModelRepository(
        models_directory=config.models_directory,
        models_index_filepath=config.models_index_filepath
    )

    main_app = MainApplicationWindow(
        league_repository=league_repository,
        model_repository=model_repository,
        themes_dict=config.themes_dict,
        app_title=config.app_title,
        help_url_links=config.help_url_links
    )
    main_app.open()


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')

    main()
