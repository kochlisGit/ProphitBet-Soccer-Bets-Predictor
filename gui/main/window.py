import variables
from database.repositories.repository import LeagueRepository
from database.network import utils
from analysis.features.correlation import CorrelationAnalyzer
from analysis.features.importance import ImportanceAnalyzer
from analysis.features.classes import ClassDistributionAnalyzer
from gui.dialogs.league import LeagueCreatorDialog, LeagueLoaderDialog, LeagueDeleteDialog
from gui.analysis.correlation import CorrelationPlotter
from gui.analysis.importance import ImportancePlotter
from gui.analysis.classes import ClassDistributionPlotter
from gui.dialogs.model import TrainNNDialog, TrainRFDialog, EvaluationDialog, PredictionDialog, PredictionUpcomingDialog
from tkinter import Tk, Menu, NORMAL, DISABLED, CENTER, HORIZONTAL, VERTICAL, messagebox, BooleanVar
from tkinter.ttk import Treeview, Scrollbar, Style


class MainWindow:
    def __init__(self, repository: LeagueRepository, all_leagues: list):
        self._league_repository = repository
        self._all_leagues = all_leagues
        self._results_and_stats = None

        self._show_help_var = None

        self._league_identifier = None
        self._results_and_stats = None
        self._correlation_analyzer = None
        self._importance_analyzer = None
        self._class_distribution_analyzer = None

        self._title = 'ProphitBet - Soccer Bets Predictor'
        self._window_size = {'width': 1420, 'height': 650}
        self._treeview_font_sizes = {'header': 12, 'row': 10}

        self._window = None
        self._menubar = None
        self._treeview = None

        self._initialize_window()

    @property
    def repository(self) -> LeagueRepository:
        return self._league_repository

    @property
    def all_leagues(self) -> list:
        return self._all_leagues

    @property
    def title(self) -> str:
        return self._title

    @property
    def window_size(self) -> dict:
        return self._window_size

    def _initialize_window(self):
        self._init_window()
        self._init_menubar()
        self._init_treeview()

        messagebox.showinfo('Welcome', 'Thank you for using our product! Always Bet Responsibly :)')

    def _init_window(self):
        window = Tk()
        window.title(self._title)
        window.geometry(f"{self.window_size['width']}x{self.window_size['height']}")
        window.resizable(False, False)
        self._window = window

    def _init_menubar(self):
        menubar = Menu(self._window)

        app_menu = Menu(menubar, tearoff=0)
        app_menu.add_command(label='Close League', command=self._close_league)
        app_menu.add_command(label='Exit', command=self._window.quit)
        menubar.add_cascade(label='Application', menu=app_menu)

        leagues_menu = Menu(menubar, tearoff=0)
        leagues_menu.add_command(label='Create League', command=self._create_league)
        leagues_menu.add_command(label='Load League', command=self._load_league)
        leagues_menu.add_separator()
        leagues_menu.add_command(label='Delete League', command=self._delete_league)
        menubar.add_cascade(label='Leagues', menu=leagues_menu)

        analysis_menu = Menu(menubar, tearoff=0)
        analysis_menu.add_command(label='Correlations', command=self._show_correlation_plotter)
        analysis_menu.add_command(label='Feature Importance', command=self._show_importance_plotter)
        analysis_menu.add_command(label='Target Distribution', command=self._show_target_distribution_plotter)
        analysis_menu.add_separator()

        self._show_help_var = BooleanVar()
        analysis_menu.add_checkbutton(label='Display Help', onvalue=1, offvalue=0, variable=self._show_help_var)
        menubar.add_cascade(label='Analysis', menu=analysis_menu, state=DISABLED)

        model_menu = Menu(menubar, tearoff=0)

        train_menu = Menu(model_menu, tearoff=0)
        train_menu.add_command(label='Neural Network', command=self._train_nn)
        train_menu.add_command(label='Random Forest', command=self._train_rf)
        model_menu.add_cascade(label='Train', menu=train_menu)

        model_menu.add_command(label='Evaluate', command=self._evaluate_model)
        model_menu.add_command(label='Predict', command=self._predict_matches)
        model_menu.add_command(label='Predict Upcoming', command=self._predict_upcoming)
        menubar.add_cascade(label='Model', menu=model_menu, state=DISABLED)

        self._menubar = menubar
        self._window.config(menu=menubar)

    def _init_treeview(self):
        columns = self._league_repository.all_columns

        self._treeview = Treeview(
            self._window,
            columns=columns,
            show='headings',
            selectmode='browse',
            height=30
        )
        for column_name in columns:
            self._treeview.column(column_name, anchor=CENTER, stretch=True, width=50)
            self._treeview.heading(column_name, text=column_name, anchor=CENTER)
        self._treeview.column('Date', anchor=CENTER, stretch=True, width=100)
        self._treeview.column('Home Team', anchor=CENTER, stretch=True, width=100)
        self._treeview.column('Away Team', anchor=CENTER, stretch=True, width=100)

        style = Style()
        style.configure('Treeview.Heading', font=('Arial Bold', 10))
        style.configure('Treeview', font=('Arial', 8))
        self._treeview.grid(row=0, column=0, sticky='nsew')

        v_scroll = Scrollbar(self._window, orient=VERTICAL, command=self._treeview.yview)
        v_scroll.grid(row=0, column=1, sticky='nse')
        self._treeview.configure(yscroll=v_scroll.set)

        h_scroll = Scrollbar(self._window, orient=HORIZONTAL, command=self._treeview.xview)
        h_scroll.grid(row=1, column=0, sticky='sew')
        self._treeview.configure(xscroll=h_scroll.set)

    def _add_results_to_treeview(self):
        for item in self._treeview.get_children():
            self._treeview.delete(item)

        results_and_stats = self._results_and_stats.values.tolist()
        for i, values in enumerate(results_and_stats):
            self._treeview.insert(parent='', index=i, values=values)

    def _enable_all_menus(self):
        for name in ['Analysis', 'Model']:
            self._menubar.entryconfig(name, state=NORMAL)

    def _disable_all_menus(self):
        for name in ['Analysis', 'Model']:
            self._menubar.entryconfig(name, state=DISABLED)

    def _create_league(self):
        if not utils.check_internet_connection():
            messagebox.showerror('No Internet Connection', 'Creating a league requires internet connection.')
            return

        creator_dialog = LeagueCreatorDialog(
            master=self._window,
            league_repository=self._league_repository,
            all_leagues=self._all_leagues
        )
        creator_dialog.start()

        self._league_identifier = creator_dialog.league_identifier
        if creator_dialog.league_identifier is not None:
            self._results_and_stats = self._league_repository.compute_results_and_stats(
                league_config=self._league_repository.get_downloaded_league_configs()[self._league_identifier]
            )

            self._add_results_to_treeview()
            self._enable_all_menus()

    def _load_league(self):
        if len(self._league_repository.get_downloaded_league_configs()) == 0:
            messagebox.showerror(
                'No Leagues Found', 'ERROR: No downloaded leagues were found. Try creating a league first.'
            )
            return

        load_dialog = LeagueLoaderDialog(self._window, self._league_repository)
        load_dialog.start()

        self._league_identifier = load_dialog.league_identifier
        if load_dialog.league_identifier is not None:
            self._results_and_stats = self._league_repository.compute_results_and_stats(
                league_config=self._league_repository.get_downloaded_league_configs()[self._league_identifier]
            )

            self._add_results_to_treeview()
            self._enable_all_menus()

    def _close_league(self):
        self._league_identifier = None
        self._results_and_stats = None
        self._correlation_analyzer = None
        self._importance_analyzer = None
        self._class_distribution_analyzer = None

        for item in self._treeview.get_children():
            self._treeview.delete(item)

        self._disable_all_menus()

    def _delete_league(self):
        if len(self._league_repository.get_downloaded_league_configs()) == 0:
            messagebox.showerror(
                'No Leagues Found', 'ERROR: No downloaded leagues were found to delete. Repository is empty.'
            )
            return

        LeagueDeleteDialog(self._window, self._league_repository, self._league_identifier).start()

    def _show_correlation_plotter(self):
        if self._correlation_analyzer is None:
            self._correlation_analyzer = CorrelationAnalyzer(results_and_stats=self._results_and_stats)

        CorrelationPlotter(self._window, self._correlation_analyzer, self._show_help_var.get()).open()

    def _show_importance_plotter(self):
        if self._importance_analyzer is None:
            self._importance_analyzer = ImportanceAnalyzer(results_and_stats=self._results_and_stats)

        ImportancePlotter(self._window, self._importance_analyzer, self._show_help_var.get()).open()

    def _show_target_distribution_plotter(self):
        if self._class_distribution_analyzer is None:
            self._class_distribution_analyzer = ClassDistributionAnalyzer(results_and_stats=self._results_and_stats)

        ClassDistributionPlotter(self._window, self._class_distribution_analyzer, self._show_help_var.get()).open()

    def _train_nn(self):
        TrainNNDialog(
            master=self._window,
            checkpoint_path=variables.checkpoint_directory,
            league_identifier=self._league_identifier,
            results_and_stats=self._results_and_stats
        ).start()

    def _train_rf(self):
        TrainRFDialog(
            master=self._window,
            checkpoint_path=variables.checkpoint_directory,
            league_identifier=self._league_identifier,
            results_and_stats=self._results_and_stats
        ).start()

    def _evaluate_model(self):
        models = self._league_repository.get_saved_models(league_identifier=self._league_identifier)
        if not models:
            messagebox.showerror('showerror', 'ERROR: No trained models where found for this league.')
            return

        EvaluationDialog(
            master=self._window,
            models=models,
            results_and_stats=self._results_and_stats,
            repository_basic_columns=self.repository.basic_columns
        ).start()

    def _predict_matches(self):
        models = self._league_repository.get_saved_models(league_identifier=self._league_identifier)
        if not models:
            messagebox.showerror('showerror', 'ERROR: No trained models where found for this league.')
            return

        PredictionDialog(
            master=self._window,
            models=models,
            results_and_stats=self._results_and_stats
        ).start()

    def _predict_upcoming(self):
        if not utils.check_internet_connection():
            messagebox.showerror('No Internet Connection', 'Creating a league requires internet connection.')
            return

        models = self._league_repository.get_saved_models(league_identifier=self._league_identifier)

        if not models:
            messagebox.showerror('showerror', 'ERROR: No trained models where found for this league.')
            return

        messagebox.showinfo(
            'Download Fixtures',
            'The app will open your web browser and redirect you to "www.footystats.org", where you can download the '
            'upcoming fixtures. Once the page is loaded, press "CTRL + S" or right click and select the option '
            '"Save as", to download the HTML page. Then, click on the button to select the path of the downloaded page.'
        )

        utils.load_page(self.repository.get_downloaded_league_configs()[self._league_identifier].league.fixtures_url)

        PredictionUpcomingDialog(
            master=self._window,
            models=models,
            results_and_stats=self._results_and_stats
        ).start()

    def open(self):
        self._window.mainloop()
