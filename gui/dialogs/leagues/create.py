from tkinter import messagebox, StringVar, BooleanVar, IntVar, END
from tkinter.ttk import Label, Combobox, Checkbutton, Entry, Button
from database.entities.leagues.league import LeagueConfig
from database.network.netutils import check_internet_connection
from database.repositories.league import LeagueRepository
from gui.dialogs.dialog import Dialog
from gui.task import TaskDialog
from gui.widgets.intslider import IntSlider
from gui.widgets.utils import create_tooltip_btn, validate_id_entry


class CreateLeagueDialog(Dialog):
    def __init__(self, root, league_repository: LeagueRepository):
        self._league_repository = league_repository
        self._all_leagues_dict = league_repository.all_leagues_dict

        self._cb_country_values = list(self._all_leagues_dict.keys())
        for i, config_list in enumerate(self._all_leagues_dict.values()):
            self._cb_country_values[i] += f' ({config_list[0].category})'

        self._odd_features = ['1', 'X', '2']
        self._home_features = ['HW', 'HL', 'HGF', 'HGA', 'HWGD', 'HLGD', 'HW%', 'HL%']
        self._away_features = ['AW', 'AL', 'AGF', 'AGA', 'AWGD', 'ALGD', 'AW%', 'AL%']
        self._matches_df = None
        self._league_config = None

        self._selected_league_cb = None
        self._league_id_entry = None
        self._year_star_cb = None
        self._selected_country_var = StringVar()
        self._league_id_var = StringVar()
        self._match_history_window_var = IntVar(value=3)
        self._goal_diff_margin_var = IntVar(value=3)
        self._odd_vars = {col: BooleanVar(value=True) for col in self._odd_features}
        self._home_vars = {col: BooleanVar(value=True) for col in self._home_features}
        self._away_vars = {col: BooleanVar(value=True) for col in self._away_features}
        self._year_start_var = IntVar(value=2015)
        self._all_vars = {**self._odd_vars, **self._home_vars, **self._away_vars}

        super().__init__(root=root, title='Create League', window_size=self._compute_required_window_size())

    def _compute_required_window_size(self) -> dict[str, int]:
        x = 400
        y = len(self._home_features)*60 + 220
        return {'width': x, 'height': y}

    def _create_widgets(self):
        Label(self.window, text='--- League Settings ---', font=('Arial', 14)).place(x=100, y=10)

        Label(self.window, text='Select Country:', font=('Arial', 14)).place(x=15, y=45)
        country_cb = Combobox(
            self.window,
            values=self._cb_country_values,
            width=21,
            font=('Arial', 10),
            state='readonly',
            textvariable=self._selected_country_var
        )
        country_cb.current(0)
        country_cb.bind('<<ComboboxSelected>>', self._adjust_league_settings)
        country_cb.place(x=165, y=50)

        Label(self.window, text='Select League:', font=('Arial', 14)).place(x=15, y=85)
        self._selected_league_cb = Combobox(
            self.window,
            width=21,
            font=('Arial', 10),
            state='readonly'
        )
        self._selected_league_cb.bind('<<ComboboxSelected>>', self._adjust_stats_settings)
        self._selected_league_cb.place(x=165, y=90)

        Label(self.window, text='League ID:', font=('Arial', 14)).place(x=15, y=125)
        self._league_id_entry = Entry(
            self.window,
            width=28,
            font=('Arial', 10),
            textvariable=self._league_id_var
        )
        self._league_id_entry.place(x=135, y=130)
        create_tooltip_btn(
            root=self.window,
            text='Identifier (ID) of this league. Each league should have unique ID.'
        ).place(x=350, y=130)

        Label(self.window, text='--- Stats Settings ---', font=('Arial', 14)).place(x=110, y=175)

        Label(self.window, text='Match History Window:', font=('Arial', 14)).place(x=15, y=215)
        IntSlider(

            self.window,
            from_=3,
            to=5,
            variable=self._match_history_window_var,
            compound='bottom'
        ).place(x=220, y=215)
        create_tooltip_btn(
            root=self.window,
            text='Number of past matches to examine to compute team stats'
        ).place(x=330, y=215)

        Label(self.window, text='Goal Margin Diff:', font=('Arial', 14)).place(x=15, y=255)
        IntSlider(
            self.window,
            from_=2,
            to=3,
            variable=self._goal_diff_margin_var,
            compound='bottom'
        ).place(x=180, y=255)
        create_tooltip_btn(
            root=self.window,
            text='Goal Difference Margin to compute HGD, HGA, AGD, AGA stats'
        ).place(x=290, y=255)

        Label(self.window, text='--- Stats Columns ---', font=('Arial', 14)).place(x=100, y=300)
        Checkbutton(
            self.window,
            text='Odd-1',
            onvalue=True,
            offvalue=False,
            variable=self._odd_vars['1']
        ).place(x=70, y=340)
        Checkbutton(
            self.window,
            text='Odd-X',
            onvalue=True,
            offvalue=False,
            variable=self._odd_vars['X']
        ).place(x=170, y=340)
        Checkbutton(
            self.window,
            text='Odd-2',
            onvalue=True,
            offvalue=False,
            variable=self._odd_vars['2']
        ).place(x=270, y=340)

        for i, col in enumerate(self._home_features):
            Checkbutton(
                self.window,
                text=col,
                onvalue=True,
                offvalue=False,
                variable=self._home_vars[col]
            ).place(x=90, y=370 + i*30)

        for i, col in enumerate(self._away_features):
            Checkbutton(
                self.window,
                text=col,
                onvalue=True,
                offvalue=False,
                variable=self._away_vars[col]
            ).place(x=240, y=370 + i*30)

        Label(self.window, text='Start Year:', font=('Arial', 14)).place(x=80, y=self.window_size['height'] - 85)
        self._year_star_cb = Combobox(
            self.window,
            width=10,
            font=('Arial', 10),
            state='readonly',
            textvariable=self._year_start_var
        )
        self._year_star_cb.place(x=190, y=self.window_size['height'] - 80)

        Button(
            self.window,
            text='Create League',
            command=self._create_league
        ).place(x=150, y=self.window_size['height'] - 40)

    def _adjust_league_settings(self, event):
        country = self._selected_country_var.get().split(' ')[0]
        self._selected_league_cb['values'] = [league.name for league in self._all_leagues_dict[country]]
        self._selected_league_cb.current(0)
        self._league_id_entry.delete(0, END)
        self._league_id_entry.insert(0, f'{country}-{self._all_leagues_dict[country][0].name}-0')

        self._adjust_stats_settings(event=event)

    def _adjust_stats_settings(self, event):
        country = self._selected_country_var.get().split(' ')[0]
        league_config = self._all_leagues_dict[country][self._selected_league_cb.current()]
        year_start = league_config.year_start
        self._year_star_cb['values'] = [year_start + i for i in range(0, 2020-year_start + 1)]
        self._year_start_var.set(value=2015)

    def _create_league(self):
        if not check_internet_connection():
            messagebox.showerror(
                parent=self.window,
                title='No Internet Connection',
                message=f'Could not connect to football-data server. Check your internet connection.'
            )
        else:
            if validate_id_entry(parent=self._window, text=self._league_id_var.get()):
                league_id = self._league_id_var.get()

                if league_id not in self._league_repository.get_created_leagues():
                    country = self._selected_country_var.get().split(' ')[0]
                    features = [col for col, var in self._all_vars.items() if var.get()]
                    league = self._all_leagues_dict[country][self._selected_league_cb.current()]
                    league.year_start = self._year_start_var.get()

                    league_config = LeagueConfig(
                        league_id=league_id,
                        league=league,
                        match_history_window=self._match_history_window_var.get(),
                        goal_diff_margin=self._goal_diff_margin_var.get(),
                        features=features
                    )
                    self._matches_df = TaskDialog(
                        master=self.window,
                        title=self.title,
                        task=self._league_repository.create_league,
                        args=(league_config,)
                    ).start()

                    if self._matches_df is not None:
                        self._league_config = league_config
                    else:
                        messagebox.showerror(
                            parent=self.window,
                            title='Failed',
                            message='Failed to create league. Check command line (cmd) for more details.'
                        )

                    self.close()
                else:
                    messagebox.showerror(
                        parent=self.window,
                        title='League-ID exists',
                        message=f'A league with ID: "{league_id}" already exists. Enter a unique ID.'
                    )

    def _init_dialog(self):
        self._window.bind('<Return>', self._get_dialog_result)
        self._adjust_league_settings(event=None)

    def _get_dialog_result(self, event=None) -> tuple:
        return self._matches_df, self._league_config
