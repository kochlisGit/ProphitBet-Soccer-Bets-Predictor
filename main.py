# repo = LeagueRepository()
# data = repo.read_league_results_and_stats('ENG')
# columns = repo.basic_columns + repo.stats_columns
#
# anal = CorrelationAnalyzer(data, columns)
# plotter = CorrelationPlotter(anal)
# anal = ImportanceAnalyzer(data, columns)
# plotter = ImportancePlotter(anal)
# plotter.mainloop()


from windows.main import LeagueWindow
from database.repository import LeagueRepository

repository = LeagueRepository()
window = LeagueWindow(repository)
window.open()
