from database.repositories.league import LeagueRepository
from database.repositories.model import ModelRepository
from gui.main.application import MainApplicationWindow
import variables


def main():
    league_repo = LeagueRepository(
        available_leagues_filepath=variables.available_leagues_filepath,
        saved_leagues_directory=variables.saved_leagues_directory
    )
    model_repo = ModelRepository(models_checkpoint_directory=variables.models_checkpoint_directory)
    main_app = MainApplicationWindow(
        league_repository=league_repo,
        model_repository=model_repo,
        random_seed=variables.random_seed
    )

    while main_app.restart:
        main_app.open()


if __name__ == "__main__":
    main()
