import tkinter as tk
import webbrowser
from tkinter import Tk, ttk, messagebox, Menu, StringVar
from tkinter.messagebox import askyesno
from database.repositories.league import LeagueRepository
from database.repositories.model import ModelRepository
from gui.dialogs import analysis
from gui.dialogs import leagues
from gui.dialogs.models import train
from gui.dialogs.models.delete import DeleteModelDialog
from gui.dialogs.models.evaluate import EvaluationDialog
from gui.dialogs.models.predict import PredictMatchesDialog
from gui.dialogs.models.fixture import PredictFixturesDialog


class MainApplicationWindow:
    def __init__(
            self,
            league_repository: LeagueRepository,
            model_repository: ModelRepository,
            app_title: str,
            themes_dict: dict[str, str],
            help_url_links: dict[str, dict[str, str]]
    ):
        self._league_repository = league_repository
        self._model_repository = model_repository
        self._app_title = app_title
        self._original_title = app_title
        self._themes_dict = themes_dict
        self._help_url_links = help_url_links

        self._root = None
        self._menubar = None
        self._treeview = None
        self._theme_name_var = None

        self._window_width = 1200
        self._window_height = 800
        self._theme_names = list(self._themes_dict.keys())
        self._matches_df = None
        self._league_config = None
        self._has_shown_welcome = False
        self._has_shown_theme_hint = False

    def _create_window(self):
        root = Tk()
        root.title(self._app_title)
        root.geometry(f'{self._window_width}x{self._window_height}')
        root.resizable(False, False)
        self._root = root

    def _create_menubar(self):
        menubar = Menu(self._root)

        app_menu = Menu(menubar, tearoff=0)
        app_menu.add_command(label='Create League', command=self._create_league)
        app_menu.add_command(label='Load League', command=self._load_league)
        app_menu.add_command(label='Close League', command=self._close_league)
        app_menu.add_separator()
        app_menu.add_command(label='Delete Leagues', command=self._delete_leagues)
        app_menu.add_separator()
        app_menu.add_command(label='Restart', command=self.restart)
        app_menu.add_command(label='Exit', command=self.quit)
        menubar.add_cascade(label='Application', menu=app_menu)

        analysis_menu = Menu(menubar, tearoff=0)
        analysis_menu.add_command(label='Targets', command=self._analyze_targets)
        analysis_menu.add_command(label='Correlation', command=self._analyze_correlations)
        analysis_menu.add_command(label='Variance', command=self._analyze_variance)
        analysis_menu.add_command(label='Importance', command=self._analyze_importance)
        menubar.add_cascade(label='Analysis', menu=analysis_menu, state='disabled')

        model_menu = Menu(menubar, tearoff=0)
        train_menu = Menu(menubar, tearoff=0)
        model_menu.add_cascade(label='Train', menu=train_menu)
        train_menu.add_command(label='Decision Tree', command=self._train_dt)
        train_menu.add_command(label='XGBoost', command=self._train_xgb)
        train_menu.add_command(label='KNN', command=self._train_knn)
        train_menu.add_command(label='Logistic Regression', command=self._train_lr)
        train_menu.add_command(label='Naive Bayes', command=self._train_nb)
        train_menu.add_command(label='Neural Network', command=self._train_nn)
        train_menu.add_command(label='Random Forest', command=self._train_rf)
        train_menu.add_command(label='SVM', command=self._train_svm)
        train_menu.add_separator()
        train_menu.add_command(label='Voting Model', command=self._train_voting_model)
        model_menu.add_command(label='Evaluate', command=self._evaluate)
        model_menu.add_command(label='Predict Matches', command=self._predict)
        model_menu.add_command(label='Predict Fixture', command=self._predict_fixture)
        model_menu.add_separator()
        model_menu.add_command(label='Delete Models', command=self._delete_models)
        menubar.add_cascade(label='Model', menu=model_menu, state='disabled')

        self._theme_name_var = StringVar(master=self._root, value=self._theme_names[0])
        theme_menu = Menu(menubar, tearoff=0)
        for theme_name in self._theme_names:
            theme_menu.add_radiobutton(
                label=theme_name,
                value=theme_name,
                variable=self._theme_name_var,
                command=self._change_theme
            )
        menubar.add_cascade(label='Theme', menu=theme_menu)

        help_menu = Menu(menubar, tearoff=0)
        for help_topic, topic_dict in self._help_url_links.items():
            topic_menu = Menu(menubar, tearoff=0)
            for topic, url in topic_dict.items():
                topic_menu.add_command(label=topic, command=lambda u=url: webbrowser.open(url=u))
            help_menu.add_cascade(label=help_topic, menu=topic_menu)
        help_menu.add_separator()
        help_menu.add_command(label='Submit Bug', command=lambda: webbrowser.open(url='https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/issues/new'))
        help_menu.add_command(label='Donate', command=lambda: webbrowser.open(url='https://www.paypal.com/donate/?hosted_button_id=AK3SEFDGVAWFE'))
        menubar.add_cascade(label='Help', menu=help_menu)

        self._root.config(menu=menubar)
        self._menubar = menubar

    def _enable_league_menus(self):
        self._menubar.entryconfig('Analysis', state='normal')
        self._menubar.entryconfig('Model', state='normal')

    def _disable_league_menus(self):
        self._menubar.entryconfig('Analysis', state='disabled')
        self._menubar.entryconfig('Model', state='disabled')

    def _change_theme(self):
        theme_name = self._theme_name_var.get()

        if not self._has_shown_theme_hint:
            self._has_shown_theme_hint = True
            messagebox.showwarning(
                parent=self._root,
                title='Theme Bug',
                message=
                    'Each theme can only be set once, due to known TKinter bug issue. '
                    'To select this theme again, you have to restart the application.'
            )

        try:
            if theme_name == 'winnative':
                return
            else:
                self._root.tk.call('source', self._themes_dict[theme_name])
                ttk.Style(self._root).theme_use(theme_name)
        except Exception as e:
            messagebox.showerror(
                parent=self._root,
                title=f'Cannot change theme',
                message=f'You cannot select the same theme due to known TKinter bug: {e}. '
                        'You must first restart the application.'
            )

    def _construct_treeview(self):
        columns = self._matches_df.columns.tolist()
        columns.insert(0, 'Index')
        items = self._matches_df.values.tolist()

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

        self._treeview = treeview

    def _load_league_matches(self):
        self._construct_treeview()
        self._enable_league_menus()
        self._root.title(self._league_config.league_id)

    def _create_league(self):
        if self._matches_df is not None:
            messagebox.showerror(parent=self._root, title='Current League Open', message='Close current league in order to create a new one.')
            return

        self._matches_df, self._league_config = leagues.CreateLeagueDialog(
            root=self._root,
            league_repository=self._league_repository
        ).open()

        if self._matches_df is not None:
            self._load_league_matches()

    def _load_league(self):
        if self._matches_df is not None:
            messagebox.showerror(parent=self._root, title='Current League Open', message='Close current league in order to load a new one.')
            return

        self._matches_df, self._league_config = leagues.LoadLeagueDialog(
            root=self._root,
            league_repository=self._league_repository
        ).open()

        if self._matches_df is not None:
            self._load_league_matches()

    def _close_league(self):
        if self._matches_df is None:
            return

        self._treeview.destroy()
        self._treeview = None
        self._disable_league_menus()
        self._root.title(self._original_title)

        self._matches_df = None
        self._league_config = None

    def _delete_leagues(self):
        leagues.DeleteLeagueDialog(
            root=self._root,
            league_repository=self._league_repository,
            model_repository=self._model_repository,
            current_league_id=None if self._league_config is None else self._league_config.league_id
        ).open()

    def _analyze_targets(self):
        analysis.targets.TargetPlotter(root=self._root, matches_df=self._matches_df).open_and_wait()

    def _analyze_correlations(self):
        analysis.correlation.CorrelationPlotter(root=self._root, matches_df=self._matches_df).open_and_wait()

    def _analyze_variance(self):
        analysis.variance.VariancePlotter(root=self._root, matches_df=self._matches_df).open_and_wait()

    def _analyze_importance(self):
        analysis.importance.ImportancePlotter(root=self._root, matches_df=self._matches_df).open_and_wait()

    def _train_dt(self):
        train.DecisionTreeTrainDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _train_xgb(self):
        train.ExtremeBoostingTrainDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _train_knn(self):
        train.KNearestNeighborsTrainDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _train_lr(self):
        train.LogisticRegressionTrainDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _train_nb(self):
        train.NaiveBayesTrainDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _train_nn(self):
        train.NeuralNetworkTrainDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _train_rf(self):
        train.RandomForestTrainDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _train_svm(self):
        train.SupportVectorMachineTrainDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _train_voting_model(self):
        train.VotingModelDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _evaluate(self):
        if len(self._model_repository.get_model_configs(league_id=self._league_config.league_id)) == 0:
            messagebox.showerror(
                parent=self._root,
                title='No Trained Models',
                message='No trained models have been found.'
            )
            return

        EvaluationDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _predict(self):
        if len(self._model_repository.get_model_configs(league_id=self._league_config.league_id)) == 0:
            messagebox.showerror(
                parent=self._root,
                title='No Trained Models',
                message='No trained models have been found.'
            )
            return

        PredictMatchesDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _predict_fixture(self):
        if len(self._model_repository.get_model_configs(league_id=self._league_config.league_id)) == 0:
            messagebox.showerror(
                parent=self._root,
                title='No Trained Models',
                message='No trained models have been found.'
            )
            return

        PredictFixturesDialog(
            root=self._root,
            matches_df=self._matches_df,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def _delete_models(self):
        if len(self._model_repository.get_model_configs(league_id=self._league_config.league_id)) == 0:
            messagebox.showerror(
                parent=self._root,
                title='No Trained Models',
                message='No trained models have been found.'
            )
            return

        DeleteModelDialog(
            root=self._root,
            league_config=self._league_config,
            model_repository=self._model_repository
        ).open_and_wait()

    def open(self):
        self._create_window()
        self._create_menubar()

        if not self._has_shown_welcome:
            self._has_shown_welcome = True
            messagebox.showinfo(
                parent=self._root,
                title='Welcome',
                message='Thank you for using our product! Always Bet Responsibly :)'
            )

        self._root.mainloop()

    def restart(self):
        self.quit()
        self.open()

    def quit(self):
        if self._league_config is None:
            quit_app = True
        else:
            quit_app = askyesno(
                parent=self._root,
                title=f'Open League',
                message=f'League {self._league_config.league_id} is open. Are you sure uou want to exit?'
            )

        if quit_app:
            self._matches_df = None
            self._league_config = None
            self._root.destroy()
            self._root.quit()
