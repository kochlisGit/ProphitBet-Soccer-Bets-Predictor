from windows.dialogs.dialog import Dialog
from windows.dialogs.task import TaskLoaderDialog
from tkinter import messagebox, StringVar, END
from tkinter.ttk import Label, Combobox, Entry, Button
from database.leagues import LeagueConfig
import threading


class LeagueCreatorDialog(Dialog):
    def __init__(self, master, league_repository):
        self._league_repository = league_repository

        self._title = 'Creating League'
        self._window_sizes = {'width': 500, 'height': 300}

        self._default_selected_league = 0
        self._default_last_n_matches = 3
        self._default_goal_diff_margin = 2

        self.directory_path = None

        self._save_entry = None
        self._league_var = StringVar()
        self._save_var = StringVar()
        self._matches_var = StringVar()
        self._goals_var = StringVar()

        super().__init__(master)

    def _initialize(self):
        self._initialize_window()
        self._initialize_form()

    def _initialize_window(self):
        self._window.title(self._title)
        self._window.geometry('{}x{}'.format(self._window_sizes['width'], self._window_sizes['height']))
        self._window.resizable(False, False)

    def _initialize_form(self):
        leagues = list(self._league_repository.get_all_leagues().keys())

        Label(self._window, text='Select League', font=('Arial', 16)).place(x=20, y=20)
        Label(self._window, text='Save League As', font=('Arial', 16)).place(x=20, y=80)
        Label(self._window, text='Last N Matches Lookup', font=('Arial', 16)).place(x=20, y=140)
        Label(self._window, text='Goal Diff Margin', font=('Arial', 16)).place(x=20, y=200)

        leagues_cb = Combobox(self._window, width=15, font=('Arial', 10), textvariable=self._league_var)
        leagues_cb['values'] = leagues
        leagues_cb.current(self._default_selected_league)
        leagues_cb.place(x=325, y=20)

        self._save_entry = Entry(self._window, width=18, font=('Arial', 10), textvariable=self._save_var)
        self._save_entry.insert(0, leagues[self._default_selected_league])
        self._save_entry.place(x=325, y=80)

        matches_entry = Entry(self._window, width=10, font=('Arial', 10), textvariable=self._matches_var)
        matches_entry.insert(0, self._default_last_n_matches)
        matches_entry.place(x=325, y=140)

        goals_entry = Entry(self._window, width=10, font=('Arial', 10), textvariable=self._goals_var)
        goals_entry.insert(0, self._default_goal_diff_margin)
        goals_entry.place(x=325, y=200)

        Button(self._window, text='Create', command=self._create_event).place(x=140, y=260)
        Button(self._window, text='Cancel', command=self.exit).place(x=260, y=260)

        leagues_cb.bind('<<ComboboxSelected>>', self._on_cb_selection_callback)

    def _on_cb_selection_callback(self, event):
        self._save_entry.delete(0, END)
        self._save_entry.insert(0, self._league_var.get())

    def _validate_form(self):
        directory_path = self._save_entry.get()
        last_n_matches = self._matches_var.get()
        goal_diff_margin = self._goals_var.get()

        if any(not c.isalnum() for c in directory_path) or directory_path[0].isdigit():
            return 'League directory must not contain special characters and must not start with a number'
        if not last_n_matches.isdigit():
            return 'Last N Matches should be a number'
        if not goal_diff_margin.isdigit():
            return 'Goal Difference Margin should be a number'
        return 'Valid'

    def _create_event(self):
        form_validation_result = self._validate_form()

        if form_validation_result == 'Valid':
            self.directory_path = self._save_entry.get()

            task_dialog = TaskLoaderDialog(self._window, self._title)
            task_thread = threading.Thread(target=self._create_league, args=(task_dialog,))
            task_thread.start()
            task_dialog.start()
            self.exit()
        else:
            messagebox.showerror('showerror', 'ERROR: ' + form_validation_result)

    def _create_league(self, task_dialog: TaskLoaderDialog):
        league_name = self._league_var.get()
        directory_path = self._save_entry.get()
        last_n_matches = int(self._matches_var.get())
        goal_diff_margin = int(self._goals_var.get())
        league_config = LeagueConfig.generate_league_config(
            league_name,
            directory_path,
            last_n_matches,
            goal_diff_margin
        )

        self._league_repository.download_league(league_config)
        task_dialog.exit()

class LeagueLoaderDialog(Dialog):
    def __init__(self, master, league_repository):
        self._league_repository = league_repository

        self._title = 'Loading League'
        self._window_sizes = {'width': 350, 'height': 120}

        self._task_thread = None
        self._save_var = StringVar()

        self.results_and_stats = None
        self.directory_path = None

        super().__init__(master)

    def _initialize(self):
        self._initialize_window()
        self._initialize_form()

    def _initialize_window(self):
        self._window.title(self._title)
        self._window.geometry('{}x{}'.format(self._window_sizes['width'], self._window_sizes['height']))
        self._window.resizable(False, False)

    def _initialize_form(self):
        downloaded_leagues = self._league_repository.get_downloaded_leagues()

        Label(self._window, text='Select League', font=('Arial', 16)).place(x=20, y=20)
        leagues_cb = Combobox(self._window, width=15, font=('Arial', 10), textvariable=self._save_var)

        leagues_cb['values'] = downloaded_leagues
        leagues_cb.current(0)
        leagues_cb.place(x=180, y=25)

        Button(self._window, width=20, text='Load', command=self._load_event).place(x=110, y=80)

    def _load_event(self):
        self.directory_path = self._save_var.get()

        task_dialog = TaskLoaderDialog(self._window, self._title)
        task_thread = threading.Thread(target=self._load_league, args=(self.directory_path, task_dialog))
        task_thread.start()
        task_dialog.start()
        self.exit()

    def _load_league(self, directory_path, task_dialog: TaskLoaderDialog):
        self._league_repository.update_league(directory_path)
        self.results_and_stats = self._league_repository.read_league_results_and_stats(directory_path)
        task_dialog.exit()


class LeagueDeleteDialog(Dialog):
    def __init__(self, master, league_repository, open_league_dir):
        self._league_repository = league_repository
        self._open_league_dir = open_league_dir

        self._title = 'Deleting League'
        self._window_sizes = {'width': 350, 'height': 120}

        self._task_thread = None
        self._save_var = StringVar()

        super().__init__(master)

    def _initialize(self):
        self._initialize_window()
        self._initialize_form()

    def _initialize_window(self):
        self._window.title(self._title)
        self._window.geometry('{}x{}'.format(self._window_sizes['width'], self._window_sizes['height']))
        self._window.resizable(False, False)

    def _initialize_form(self):
        downloaded_leagues = self._league_repository.get_downloaded_leagues()

        Label(self._window, text='Select League', font=('Arial', 16)).place(x=20, y=20)
        leagues_cb = Combobox(self._window, width=15, font=('Arial', 10), textvariable=self._save_var)

        leagues_cb['values'] = downloaded_leagues
        leagues_cb.current(0)
        leagues_cb.place(x=180, y=25)

        Button(self._window, width=10, text='Delete', command=self._delete_event).place(x=80, y=80)
        Button(self._window, width=10, text='Cancel', command=self.exit).place(x=200, y=80)

    def _delete_event(self):
        directory_path = self._save_var.get()

        if self._open_league_dir == directory_path:
            messagebox.showerror('showerror', 'ERROR: League is currently open!')
        else:
            self._league_repository.delete_league(directory_path)
            self.exit()
