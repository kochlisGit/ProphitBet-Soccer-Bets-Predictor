from gui.main.window import MainWindow
from database.repositories.repository import LeagueRepository
from database.reader import LeagueReader
import variables


def main():
    all_leagues = LeagueReader.read_all_leagues(leagues_filepath=variables.leagues_list_filepath)
    repository = LeagueRepository(
        repository_directory=variables.repository_directory,
        checkpoint_directory=variables.checkpoint_directory
    )
    main_window = MainWindow(repository=repository, all_leagues=all_leagues)
    main_window.open()


if __name__ == '__main__':
    main()
