import tkinter as tk
import webbrowser
from tkinter import Tk, ttk, messagebox, Menu, BooleanVar, StringVar
from database.repositories.league import LeagueRepository
from database.repositories.model import ModelRepository
from gui.analysis.features.classes import ClassDistributionPlotter
from gui.analysis.features.correlation import CorrelationPlotter
from gui.analysis.features.importance import ImportancePlotter
from gui.dialogs.league import CreateLeagueDialog, LoadLeagueDialog, DeleteLeagueDialog
from gui.dialogs.model.evaluation import EvaluationDialog
from gui.dialogs.model.fixture import FixturesDialog
from gui.dialogs.model.predictions import PredictionDialog
from gui.dialogs.model.training import TrainCustomNNDialog, TrainCustomRFDialog
from gui.dialogs.model.tuning import TuningNNDialog, TuningRFDialog


class MainApplicationWindow:
    def __init__(self, league_repository: LeagueRepository, model_repository: ModelRepository, random_seed: int):
        self._league_repository = league_repository
        self._model_repository = model_repository
        self._random_seed = random_seed

        self._root = None
        self._menubar = None

        self._theme_name_var = None
        self._show_hints_var = None

        self._title = 'ProphitBet'
        self._window_width = 1200
        self._window_height = 700
        self._has_shown_welcome_message = False
        self._has_shown_change_them_hint = False
        self._restart = True

        self._matches_df = None
        self._open_league_name = None
        self._open_league = None

    @property
    def restart(self) -> bool:
        return self._restart

    def open(self):
        self._initialize_user_interface()

        if not self._has_shown_welcome_message:
            messagebox.showinfo('Welcome', 'Thank you for using our product! Always Bet Responsibly :)')
            self._has_shown_welcome_message = True

        self._restart = False
        self._root.mainloop()

    def _initialize_user_interface(self):
        self._create_main_window()
        self._add_menubar()

    def _create_main_window(self):
        root = Tk()
        root.title(self._title)
        root.geometry(f'{self._window_width}x{self._window_height}')
        root.resizable(False, False)
        self._root = root

    def _add_menubar(self):
        menubar = Menu(self._root)

        app_menu = Menu(menubar, tearoff=0)
        app_menu.add_command(label='Create League', command=self._create_league)
        app_menu.add_command(label='Load League', command=self._load_league)
        app_menu.add_command(label='Delete League', command=self._delete_league)
        app_menu.add_separator()
        app_menu.add_command(label='Close League', command=self._close_league)
        app_menu.add_separator()
        app_menu.add_command(label='Exit', command=self._root.quit)
        menubar.add_cascade(label='Application', menu=app_menu)

        analysis_menu = Menu(menubar, tearoff=0)
        analysis_menu.add_command(label='Correlations', command=self._plot_correlations)
        analysis_menu.add_command(label='Feature Importance', command=self._plot_importance)
        analysis_menu.add_command(label='Target Distribution', command=self._plot_target_distribution)
        menubar.add_cascade(label='Analysis', menu=analysis_menu, state='disabled')

        model_menu = Menu(menubar, tearoff=0)
        train_menu = Menu(menubar, tearoff=0)
        train_menu.add_command(label='Neural Network (Auto Tuning)', command=self._tune_nn)
        train_menu.add_command(label='Neural Network (Custom)', command=self._train_custom_nn)
        train_menu.add_separator()
        train_menu.add_command(label='Random Forest (Auto Tuning)', command=self._tune_rf)
        train_menu.add_command(label='Random Forest (Custom)', command=self._train_custom_rf)
        model_menu.add_cascade(label='Train', menu=train_menu)
        model_menu.add_command(label='Evaluate', command=self._evaluate_models)
        model_menu.add_command(label='Predict Matches', command=self._predict_matches)
        model_menu.add_command(label='Predict Fixture', command=self._predict_fixture)
        menubar.add_cascade(label='Model', menu=model_menu, state='disabled')

        self._theme_name_var = StringVar(value='winnative')

        theme_menu = Menu(menubar, tearoff=0)
        theme_menu.add_radiobutton(
            label='Default', value='winnative', variable=self._theme_name_var, command=self._change_theme
        )
        theme_menu.add_radiobutton(
            label='Breeze (Light)', value='breeze-light', variable=self._theme_name_var, command=self._change_theme
        )
        theme_menu.add_radiobutton(
            label='Breeze (Dark)', value='breeze-dark', variable=self._theme_name_var, command=self._change_theme
        )
        theme_menu.add_radiobutton(
            label='Forest (Light)', value='forest-light', variable=self._theme_name_var, command=self._change_theme
        )
        theme_menu.add_radiobutton(
            label='Forest (Dark)', value='forest-dark', variable=self._theme_name_var, command=self._change_theme
        )
        menubar.add_cascade(label='Theme', menu=theme_menu)

        self._show_hints_var = BooleanVar(value=True)

        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_checkbutton(label='Display Help', onvalue=1, offvalue=0, variable=self._show_hints_var)
        help_menu.add_command(
            label='About', command=lambda: webbrowser.open('https://kochlisgit.github.io/aboutme/')
        )
        help_menu.add_separator()
        help_menu.add_command(
            label='Neural Networks', command=lambda: webbrowser.open('https://www.ibm.com/topics/neural-networks')
        )
        help_menu.add_command(
            label='Random Forests',
            command=lambda: webbrowser.open('https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/')
        )
        help_menu.add_command(
            label='Classification Metrics',
            command=lambda: webbrowser.open(
                'https://www.kdnuggets.com/2020/04/performance-evaluation-metrics-classification.html')
        )
        help_menu.add_command(
            label='Imbalanced Claases',
            command=lambda: webbrowser.open(
                'https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data')
        )
        help_menu.add_command(
            label='Correlation Analysis',
            command=lambda: webbrowser.open(
                'https://www.lri.fr/~pierres/donn%E9es/save/these/articles/lpr-queue/hall99correlationbased.pdf')
        )
        menubar.add_cascade(label='Help', menu=help_menu)

        self._root.config(menu=menubar)
        self._menubar = menubar

    def _create_league(self):
        if self._matches_df is not None:
            messagebox.showwarning(
                'League is Open', 'Another league is currently opened. Select "Close League"'
                                  'in order to be able to create another league')
            return

        self._open_league_name, self._open_league, self._matches_df = CreateLeagueDialog(
            root=self._root,
            league_repository=self._league_repository
        ).open()

        if self._matches_df is not None:
            self._construct_league_treeview(
                columns=self._matches_df.columns.values.tolist(),
                items=self._matches_df.values.tolist()
            )
            self._enable_menus()

    def _load_league(self):
        if self._matches_df is not None:
            messagebox.showwarning(
                'League is Open', 'Another league is currently opened. Select "Close League"'
                                  'in order to be able to open another league')
            return

        if len(self._league_repository.get_all_saved_leagues()) == 0:
            messagebox.showwarning(
                'No Saved League', 'No leagues have been created yet. To create one, click on "Create League" menu'
            )
            return

        self._open_league_name, self._open_league, self._matches_df = LoadLeagueDialog(
            root=self._root,
            league_repository=self._league_repository
        ).open()

        if self._matches_df is not None:
            self._construct_league_treeview(
                columns=self._matches_df.columns.values.tolist(),
                items=self._matches_df.values.tolist()
            )
            self._enable_menus()

    def _delete_league(self):
        if len(self._league_repository.get_all_saved_leagues()) == 0:
            messagebox.showwarning(
                'No Saved League', 'No leagues have been created yet, so this option is not available'
            )
            return

        DeleteLeagueDialog(
            root=self._root,
            league_repository=self._league_repository,
            open_league_name=self._open_league_name
        ).open()

    def _close_league(self):
        self._matches_df = None
        self._open_league_name = None
        self._open_league = None
        self._restart = True
        self._root.destroy()

    def _plot_target_distribution(self):
        ClassDistributionPlotter(
            root=self._root,
            matches_df=self._matches_df,
            show_help=self._show_hints_var.get()
        ).open()

    def _plot_importance(self):
        ImportancePlotter(
            root=self._root,
            matches_df=self._matches_df,
            show_help=self._show_hints_var.get()
        ).open()

    def _plot_correlations(self):
        CorrelationPlotter(
            root=self._root,
            matches_df=self._matches_df,
            show_help=self._show_hints_var.get()
        ).open()

    def _tune_nn(self):
        TuningNNDialog(
            root=self._root,
            model_repository=self._model_repository,
            league_name=self._open_league_name,
            random_seed=self._random_seed,
            matches_df=self._matches_df
        ).open()

    def _train_custom_nn(self):
        TrainCustomNNDialog(
            root=self._root,
            model_repository=self._model_repository,
            league_name=self._open_league_name,
            matches_df=self._matches_df,
            random_seed=self._random_seed
        ).open()

    def _tune_rf(self):
        TuningRFDialog(
            root=self._root,
            model_repository=self._model_repository,
            league_name=self._open_league_name,
            random_seed=self._random_seed,
            matches_df=self._matches_df
        ).open()

    def _train_custom_rf(self):
        TrainCustomRFDialog(
            root=self._root,
            model_repository=self._model_repository,
            league_name=self._open_league_name,
            matches_df=self._matches_df,
            random_seed=self._random_seed
        ).open()

    def _evaluate_models(self):
        if self._model_repository.get_all_models(league_name=self._open_league_name) is None:
            messagebox.showerror(
                'No Trained Model',
                'There are no trained models to evaluate on this league. '
                'To train a model, click on "Model/Train" menu and choose to train a model.'
            )
            return

        EvaluationDialog(
            root=self._root,
            matches_df=self._matches_df,
            model_repository=self._model_repository,
            league_name=self._open_league_name
        ).open()

    def _predict_matches(self):
        if self._model_repository.get_all_models(league_name=self._open_league_name) is None:
            messagebox.showerror(
                'No Trained Model',
                'There are no trained models to make predictions on this league. '
                'To train a model, click on "Model/Train" menu and choose to train a model.'
            )
            return

        PredictionDialog(
            root=self._root,
            matches_df=self._matches_df,
            model_repository=self._model_repository,
            league_name=self._open_league_name
        ).open()

    def _predict_fixture(self):
        if self._model_repository.get_all_models(league_name=self._open_league_name) is None:
            messagebox.showerror(
                'No Trained Model',
                'There are no trained models to predict fixtures of this league. '
                'To train a model, click on "Model/Train" menu and choose to train a model.'
            )
            return

        FixturesDialog(
            root=self._root,
            matches_df=self._matches_df,
            model_repository=self._model_repository,
            league_name=self._open_league_name,
            league_fixture_url=self._open_league.fixtures_url
        ).open()

    def _construct_league_treeview(self, columns: list, items: list):
        columns.insert(0, 'Index')
        treeview = ttk.Treeview(
            self._root,
            columns=columns,
            show='headings',
            selectmode='extended',
            height=30
        )
        for column_name in columns:
            treeview.column(column_name, anchor='center', stretch=True, width=60)
            treeview.heading(column_name, text=column_name, anchor='center')
        treeview.pack(expand=True, fill='both')
        treeview.column('Date', anchor='center', stretch=True, width=100)
        treeview.column('Home Team', anchor='center', stretch=True, width=120)
        treeview.column('Away Team', anchor='center', stretch=True, width=120)

        v_scroll = tk.Scrollbar(treeview, orient='vertical', command=treeview.yview)
        v_scroll.pack(side='right', fill='y')
        h_scroll = tk.Scrollbar(treeview, orient='horizontal', command=treeview.xview)
        h_scroll.pack(side='bottom', fill='x')
        treeview.config(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        for i, values in enumerate(items):
            treeview.insert(parent='', index=i, values=[i + 1] + values)

    def _enable_menus(self):
        for name in ['Analysis', 'Model']:
            self._menubar.entryconfig(name, state='normal')

    def _change_theme(self):
        theme_name = self._theme_name_var.get()

        if not self._has_shown_change_them_hint:
            messagebox.showwarning(
                'Theme Warning',
                'Due to TKinter library bug, the same theme can only be set once. '
                'To set this theme again, You have to close the current league or restart the application.'
            )
            self._has_shown_change_them_hint = True

        try:
            if theme_name == 'winnative':
                return
            elif theme_name == 'forest-light':
                self._root.tk.call('source', 'database/storage/themes/forest/forest-light.tcl')
                ttk.Style(self._root).theme_use('forest-light')
            elif theme_name == 'forest-dark':
                self._root.tk.call('source', 'database/storage/themes/forest/forest-dark.tcl')
                ttk.Style(self._root).theme_use('forest-dark')
            elif theme_name == 'breeze-light':
                self._root.tk.call('source', 'database/storage/themes/breeze/breeze/breeze.tcl')
                ttk.Style(self._root).theme_use('breeze-light')
            elif theme_name == 'breeze-dark':
                self._root.tk.call('source', 'database/storage/themes/breeze/breeze-dark/breeze-dark.tcl')
                ttk.Style(self._root).theme_use('breeze-dark')
            else:
                return
        except:
            print('Exception Error: Cannot dynamically change theme more than once, due to TKinter Bug')
