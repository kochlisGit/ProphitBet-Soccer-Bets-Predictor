from tkinter import Tk, Menu, NORMAL, DISABLED, CENTER, HORIZONTAL, VERTICAL, messagebox
from tkinter.ttk import Treeview, Scrollbar, Style
from windows.dialogs.league import LeagueCreatorDialog, LeagueLoaderDialog, LeagueDeleteDialog
from analysis.correlation import CorrelationAnalyzer
from analysis.importance import ImportanceAnalyzer
from windows.plotters.analysis.correlation import CorrelationPlotter
from windows.plotters.analysis.importance import ImportancePlotter
from windows.dialogs.model import TrainDialog, EvaluationDialog, PredictionsDialog


class LeagueWindow:
    def __init__(self, league_repository):
        self._league_repository = league_repository

        self._title = 'Soccer Bets Predictor'
        self._window_sizes = {'width': 1420, 'height': 650}
        self._treeview_font_sizes = {'header': 12, 'row': 10}

        self._open_league_dir = None
        self._results_and_stats = None
        self._correlation_analyzer = None
        self._importance_analyzer = None

        self._window = None
        self._menubar = None
        self._treeview = None

        self._initialize()

    def _initialize(self):
        self._init_window()
        self._init_menubar()
        self._init_treeview()

    def _init_window(self):
        window = Tk()
        window.title(self._title)
        window.geometry('{}x{}'.format(self._window_sizes['width'], self._window_sizes['height']))
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
        menubar.add_cascade(label='Analysis', menu=analysis_menu, state=DISABLED)

        model_menu = Menu(menubar, tearoff=0)
        model_menu.add_command(label='Train', command=self._train_model)
        model_menu.add_command(label='Evaluate', command=self._evaluate_model)
        model_menu.add_command(label='Predict', command=self._make_predictions)
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

        for i, values in enumerate(self._results_and_stats):
            self._treeview.insert(parent='', index=i, values=values)

    def _enable_all_menus(self):
        for name in ['Analysis', 'Model']:
            self._menubar.entryconfig(name, state=NORMAL)

    def _disable_all_menus(self):
        for name in ['Analysis', 'Model']:
            self._menubar.entryconfig(name, state=DISABLED)

    def _create_league(self):
        creator_dialog = LeagueCreatorDialog(self._window, self._league_repository)
        creator_dialog.start()

        if creator_dialog.directory_path is not None:
            self._results_and_stats = self._league_repository.read_league_results_and_stats(
                creator_dialog.directory_path
            )
            self._open_league_dir = creator_dialog.directory_path
            self._add_results_to_treeview()
            self._enable_all_menus()

    def _load_league(self):
        if len(self._league_repository.get_downloaded_leagues()) == 0:
            messagebox.showerror('showerror', 'ERROR: No downloaded leagues were found.')
            return

        load_dialog = LeagueLoaderDialog(self._window, self._league_repository)
        load_dialog.start()

        self._results_and_stats = load_dialog.results_and_stats
        self._open_league_dir = load_dialog.directory_path
        self._add_results_to_treeview()
        self._enable_all_menus()

    def _delete_league(self):
        if len(self._league_repository.get_downloaded_leagues()) == 0:
            messagebox.showerror('showerror', 'ERROR: No downloaded leagues were found.')
            return

        LeagueDeleteDialog(self._window, self._league_repository, self._open_league_dir).start()

    def _close_league(self):
        self._open_league_dir = None
        self._results_and_stats = None
        self._correlation_analyzer = None
        self._importance_analyzer = None

        for item in self._treeview.get_children():
            self._treeview.delete(item)

        self._disable_all_menus()

    def _show_correlation_plotter(self):
        if self._correlation_analyzer is None:
            self._correlation_analyzer = CorrelationAnalyzer(
                self._results_and_stats,
                self._league_repository.all_columns
            )

        CorrelationPlotter(self._window, self._correlation_analyzer).open()

    def _show_importance_plotter(self):
        if self._importance_analyzer is None:
            self._importance_analyzer = ImportanceAnalyzer(
                self._results_and_stats,
                self._league_repository.all_columns
            )

        ImportancePlotter(self._window, self._importance_analyzer).open()

    def _train_model(self):
        TrainDialog(
            self._window,
            self._open_league_dir,
            self._results_and_stats,
            self._league_repository.all_columns
        ).start()

    def _evaluate_model(self):
        if not self._league_repository.model_exists(self._open_league_dir):
            messagebox.showerror('showerror', 'ERROR: No trained models where found for this league.')
            return

        EvaluationDialog(
            self._window,
            self._open_league_dir,
            self._results_and_stats,
            self._league_repository.all_columns,
            self._league_repository.basic_columns
        ).start()

    def _make_predictions(self):
        if not self._league_repository.model_exists(self._open_league_dir):
            messagebox.showerror('showerror', 'ERROR: No trained models where found for this league.')
            return

        PredictionsDialog(
            self._window,
            self._open_league_dir,
            self._results_and_stats,
            self._league_repository.all_columns
        ).start()

    def open(self):
        self._window.mainloop()
