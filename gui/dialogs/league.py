import threading
import pandas as pd
from database.entities.league import League
from gui.dialogs.dialog import Dialog
from tkinter import messagebox, StringVar, BooleanVar, IntVar, END
from tkinter.ttk import Label, Combobox, Checkbutton, Entry, Button
from database.repositories.league import LeagueRepository
from gui.dialogs.task import TaskDialog


class CreateLeagueDialog(Dialog):
    def __init__(self, root, league_repository: LeagueRepository):
        super().__init__(
            root=root,
            title="Creating League",
            window_size={"width": 1100, "height": 680},
        )

        self._league_repository = league_repository
        self._all_leagues = league_repository.get_all_available_leagues()

        all_columns = league_repository.get_all_available_columns()
        self._home_columns = [col for col in all_columns if col[0] == "H"]
        self._away_columns = [col for col in all_columns if col[0] == "A"]

        self._default_last_n_matches = 3
        self._default_goal_diff_margin = 2

        self._selected_league_and_matches_df = (None, None, None)
        self._selected_league_var = StringVar()
        self._league_name_var = StringVar()
        self._last_n_matches_var = StringVar()
        self._goal_diff_margin_var = StringVar()

        self._home_columns_vars = [
            IntVar(value=1) for _ in range(len(self._home_columns))
        ]
        self._away_columns_vars = [
            IntVar(value=1) for _ in range(len(self._away_columns))
        ]

    def _initialize(self):
        all_leagues = list(self._all_leagues.values())

        Label(self.window, text="Select League", font=("Arial", 16)).place(x=20, y=15)
        Label(self.window, text="League Name", font=("Arial", 16)).place(x=20, y=50)
        Label(self.window, text="Last N Matches Lookup", font=("Arial", 16)).place(
            x=20, y=90
        )
        Label(self.window, text="Goal Diff Margin", font=("Arial", 16)).place(
            x=20, y=130
        )
        Label(self.window, text="Statistics Columns", font=("Arial", 16)).place(
            x=170, y=180
        )

        leagues_cb = Combobox(
            self.window,
            width=25,
            font=("Arial", 10),
            state="readonly",
            textvariable=self._selected_league_var,
        )
        leagues_cb["values"] = [
            f"{league.country} - {league.name}" for league in all_leagues
        ]
        leagues_cb.current(10)
        leagues_cb.place(x=325, y=15)

        league_name_entry = Entry(
            self.window,
            width=28,
            font=("Arial", 10),
            textvariable=self._league_name_var,
        )
        league_name_entry.insert(0, all_leagues[10].name.replace("-", ""))
        league_name_entry.place(x=325, y=50)

        last_n_entry = Entry(
            self.window,
            width=10,
            font=("Arial", 10),
            textvariable=self._last_n_matches_var,
        )
        last_n_entry.insert(0, f"{self._default_last_n_matches}")
        last_n_entry.place(x=325, y=90)

        goal_diff_entry = Entry(
            self.window,
            width=10,
            font=("Arial", 10),
            textvariable=self._goal_diff_margin_var,
        )
        goal_diff_entry.insert(0, f"{self._default_goal_diff_margin}")
        goal_diff_entry.place(x=325, y=130)

        for i, col in enumerate(self._home_columns):
            Checkbutton(
                self.window,
                text=col,
                variable=self._home_columns_vars[i],
                onvalue=1,
                offvalue=0,
            ).place(x=40 + i * 60, y=220)
        for i, col in enumerate(self._away_columns):
            Checkbutton(
                self.window,
                text=col,
                variable=self._away_columns_vars[i],
                onvalue=1,
                offvalue=0,
            ).place(x=40 + i * 60, y=240)

        Button(self.window, text="Create", command=self._create_league).place(
            x=175, y=300
        )
        Button(self.window, text="Cancel", command=self.close).place(x=295, y=300)

        leagues_cb.bind(
            "<<ComboboxSelected>>",
            lambda event: self._league_selection_callback(
                event=event, league_name_entry=league_name_entry
            ),
        )

    def _league_selection_callback(self, event, league_name_entry):
        league_name_entry.delete(0, END)
        league_name_entry.insert(
            0,
            self._selected_league_var.get()
            .replace(" ", "")
            .split("-", maxsplit=1)[1]
            .replace("-", ""),
        )

    def _create_league(self):
        validation_result = self._validate_form()
        league_name = self._league_name_var.get()

        if validation_result == "valid":
            if not self._league_repository.league_exists(league_name=league_name):
                task_dialog = TaskDialog(self.window, self._title)
                task_thread = threading.Thread(
                    target=self._store_league, args=(task_dialog,)
                )
                task_thread.start()
                task_dialog.open()
                self.close()
            else:
                messagebox.showerror(
                    "League Exists",
                    f'A league with name "{league_name} already exists". Choose another name league',
                )
        else:
            messagebox.showerror("Invalid Inputs", validation_result)

    def _store_league(self, task_dialog: TaskDialog):
        selected_league = self._selected_league_var.get()
        league_name = self._league_name_var.get()
        last_n_matches = int(self._last_n_matches_var.get())
        goal_diff_margin = int(self._goal_diff_margin_var.get())

        selected_league_split = selected_league.replace(" ", "").split("-", maxsplit=1)
        selected_home_columns = [
            col
            for i, col in enumerate(self._home_columns)
            if self._home_columns_vars[i].get()
        ]
        selected_away_columns = [
            col
            for i, col in enumerate(self._away_columns)
            if self._home_columns_vars[i].get()
        ]

        matches_df, league = self._league_repository.create_league(
            league=self._all_leagues[
                (selected_league_split[0], selected_league_split[1])
            ],
            last_n_matches=last_n_matches,
            goal_diff_margin=goal_diff_margin,
            statistic_columns=selected_home_columns + selected_away_columns,
            league_name=league_name,
        )

        if matches_df is None:
            messagebox.showerror(
                "Internet Connection Error",
                "Application cannot connect to the internet. Check internet connection",
            )
        else:
            self._selected_league_and_matches_df = (league_name, league, matches_df)
        task_dialog.close()

    def _validate_form(self):
        league_name = self._league_name_var.get()
        last_n_matches = self._last_n_matches_var.get()
        goal_diff_margin = self._goal_diff_margin_var.get()

        if any(not c.isalnum() for c in league_name) or league_name[0].isdigit():
            return "League directory must not contain special characters and must not start with a number"
        if not last_n_matches.isdigit() or int(last_n_matches) <= 0:
            return "Last N Matches should be a positive integer"
        if not goal_diff_margin.isdigit() or int(goal_diff_margin) <= 0:
            return "Goal Difference Margin should be a positive integer"
        return "valid"

    def _dialog_result(self) -> (str, League, pd.DataFrame) or (None, None, None):
        return self._selected_league_and_matches_df


class LoadLeagueDialog(Dialog):
    def __init__(self, root, league_repository: LeagueRepository):
        super().__init__(
            root=root, title="Load League", window_size={"width": 350, "height": 160}
        )

        self._league_repository = league_repository
        self._selected_league_and_matches_df = (None, None, None)
        self._league_name_var = StringVar()
        self._update_var = BooleanVar(value=True)

    def _initialize(self):
        Label(self.window, text="-- Select League --", font=("Arial", 16)).place(
            x=100, y=10
        )

        leagues_cb = Combobox(
            self.window,
            width=15,
            font=("Arial", 10),
            state="readonly",
            textvariable=self._league_name_var,
        )
        leagues_cb["values"] = self._league_repository.get_all_saved_leagues()
        leagues_cb.current(0)
        leagues_cb.place(x=130, y=50)

        Checkbutton(
            self.window,
            text="Update League",
            onvalue=True,
            offvalue=False,
            width=15,
            variable=self._update_var,
        ).place(x=130, y=80)

        Button(self.window, width=20, text="Load", command=self._load_league).place(
            x=110, y=125
        )

    def _load_league(self):
        task_dialog = TaskDialog(self.window, self._title)
        task_thread = threading.Thread(
            target=self._load_league_matches, args=(task_dialog,)
        )
        task_thread.start()
        task_dialog.open()
        self.close()

    def _load_league_matches(self, task_dialog: TaskDialog):
        league_name = self._league_name_var.get()
        matches_df = None
        league = None

        if self._update_var.get():
            matches_df, league = self._league_repository.update_league(
                league_name=league_name
            )

            if matches_df is None:
                messagebox.showwarning(
                    "Internet Connection Warning",
                    "Application cannot connect to the internet. Skipping League Update",
                )

        if matches_df is None:
            matches_df, league = self._league_repository.load_league(
                league_name=league_name
            )

        self._selected_league_and_matches_df = (league_name, league, matches_df)
        task_dialog.close()

    def _dialog_result(self) -> (str, League, pd.DataFrame):
        return self._selected_league_and_matches_df


class DeleteLeagueDialog(Dialog):
    def __init__(
        self, root, league_repository: LeagueRepository, open_league_name: str or None
    ):
        super().__init__(
            root=root, title="Delete League", window_size={"width": 350, "height": 120}
        )

        self._league_repository = league_repository
        self._all_league_names = self._league_repository.get_all_saved_leagues()

        self._leagues_cb = None
        self._league_name_var = StringVar()
        self._open_league_name = open_league_name

    def _initialize(self):
        Label(self.window, text="Select League", font=("Arial", 16)).place(x=20, y=20)

        leagues_cb = Combobox(
            self.window,
            width=15,
            font=("Arial", 10),
            state="readonly",
            textvariable=self._league_name_var,
        )
        leagues_cb["values"] = self._all_league_names
        leagues_cb.current(0)
        leagues_cb.place(x=180, y=25)
        self._leagues_cb = leagues_cb

        Button(self.window, width=20, text="Delete", command=self._delete_league).place(
            x=30, y=80
        )
        Button(self.window, width=20, text="Close", command=self.close).place(
            x=200, y=80
        )

    def _delete_league(self):
        task_dialog = TaskDialog(self.window, self._title)
        task_thread = threading.Thread(
            target=self._delete_league_files, args=(task_dialog,)
        )
        task_thread.start()
        task_dialog.open()

    def _delete_league_files(self, task_dialog: TaskDialog):
        league_name = self._league_name_var.get()

        if self._open_league_name is not None and league_name == self._open_league_name:
            messagebox.showerror(
                "League Open Error",
                f'Cannot delete league "{league_name}", because it is currently open. '
                f'To delete this league, select "Close League" option and try again',
            )
        else:
            self._league_repository.delete_league(league_name=league_name)
            self._all_league_names.remove(league_name)

            messagebox.showinfo(
                "League Deleted",
                f'League "{league_name}" has been successfully deleted',
            )

            if len(self._all_league_names) == 0:
                self.close()
            else:
                self._leagues_cb["values"] = self._all_league_names
                self._leagues_cb.current(0)
        task_dialog.close()

    def _dialog_result(self) -> bool:
        return True
