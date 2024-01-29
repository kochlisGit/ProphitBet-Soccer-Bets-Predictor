import os
import numpy as np
import pandas as pd
import webbrowser
import config
from tkinter import messagebox, filedialog, StringVar, IntVar
from tkinter.ttk import Treeview, Combobox, Label, Scrollbar, Button, Entry
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from fixtures.footystats.scraper import FootyStatsScraper
from fixtures.utils import match_fixture_teams
from gui.dialogs.dialog import Dialog
from gui.task import TaskDialog
from gui.widgets.utils import validate_odd_entry
from models.tasks import ClassificationTask
from preprocessing.dataset import DatasetPreprocessor


class PredictFixturesDialog(Dialog):
    def __init__(self, root, matches_df: pd.DataFrame, league_config: LeagueConfig, model_repository: ModelRepository):
        super().__init__(root=root, title='Predictions', window_size={'width': 900, 'height': 660})

        self._matches_df = matches_df.dropna().reset_index(drop=True)
        self._league_config = league_config
        self._model_repository = model_repository

        self._model_configs = model_repository.get_model_configs(league_id=league_config.league_id)
        self._dataset_preprocessor = DatasetPreprocessor()
        self._tasks = {
            task.name: task for task in [ClassificationTask.Result, ClassificationTask.Over]
            if task.name in self._model_configs and len(self._model_configs[task.name]) > 0
        }
        self._task_predictions = {
            'Result': {0: 'H', 1: 'D', 2: 'A'},
            'Over': {0: 'U(2.5)', 1: 'O(2.5)'}
        }
        self._match_columns = set(self._matches_df.columns)
        self._all_teams = sorted(self._matches_df['Home Team'].unique().tolist())
        self._treeview_columns = {
            'Result': ['Date', 'Home Team', 'Away Team', '1', 'X', '2', 'Predicted', 'Prob-H', 'Prob-D', 'Prob-A'],
            'Over': ['Date', 'Home Team', 'Away Team', '1', 'X', '2', 'Predicted', 'Prob-U(2.5)', 'Prob-O(2.5)']
        }
        self._fixture_months = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }

        self._task_var = StringVar()
        self._model_id_var = StringVar()
        self._browser_var = StringVar()
        self._month_var = StringVar()
        self._day_var = IntVar()

        self._model_cb = None
        self._treeview = None
        self._parse_btn = None
        self._predict_btn = None

    def _adjust_task(self, event):
        def adjust_models(task: str):
            self._model_cb['values'] = list(self._model_configs[task].keys())
            self._model_id_var.set(value='')

        def adjust_treeview(task: str):
            if self._treeview is not None:
                self._treeview.destroy()
                self._treeview_scroll.destroy()

            self._treeview = Treeview(
                self.window,
                show='headings',
                selectmode='extended',
                height=25
            )
            self._treeview.place(x=10, y=60)

            treeview_columns = self._treeview_columns[task]
            self._treeview['columns'] = treeview_columns
            for column_name in treeview_columns:
                self._treeview.column(column_name, anchor='center', stretch=True, width=70)
                self._treeview.heading(column_name, text=column_name, anchor='center')
            self._treeview.column('Home Team', anchor='center', stretch=True, width=100)
            self._treeview.column('Away Team', anchor='center', stretch=True, width=100)
            self._treeview.update()

            self._treeview_scroll = Scrollbar(self.window, orient='vertical', command=self._treeview.yview)
            self._treeview_scroll.place(
                x=self._treeview.winfo_reqwidth() + 30,
                y=60,
                height=self._treeview.winfo_reqheight()
            )
            self._treeview.configure(yscroll=self._treeview_scroll.set)
            self._treeview.bind('<Double-1>', self._edit_item_on_double_click_event)

        task = self._task_var.get()

        if not task:
            return

        adjust_models(task=task)
        adjust_treeview(task=task)
        self._parse_btn['state'] = 'normal'
        self._predict_btn['state'] = 'disabled'
        self._export_btn['state'] = 'disabled'

    def _edit_item_on_double_click_event(self, event):
        item = self._treeview.selection()[0]
        column_index = int(self._treeview.identify_column(event.x).replace('#', '')) - 1
        column_name = self._treeview_columns[self._task_var.get()][column_index]
        current_value = self._treeview.item(item, 'values')[column_index]

        if column_name == 'Home Team' or column_name == 'Away Team':
            team_var = StringVar(value=current_value)
            cb = Combobox(self.window, state='readonly', width=15, textvariable=team_var)
            cb['values'] = self._all_teams
            cb.place(x=50, y=50)

            def on_cb_selected(event):
                self._treeview.set(item, column=column_name, value=team_var.get())
                cb.destroy()

            cb.bind("<<ComboboxSelected>>", on_cb_selected)
        else:
            if column_name != 'Date':
                validate_float = self.window.register(validate_odd_entry)
                entry = Entry(self.window, validate='key', validatecommand=(validate_float, '%P'))
            else:
                entry = Entry(self.window)
            entry.insert(0, current_value)
            entry.place(x=50, y=50)

            def on_entry_return(event):
                self._treeview.set(item, column=column_name, value=entry.get())
                entry.destroy()
            entry.bind("<Return>", on_entry_return)

    def _add_items(self, matches_df: pd.DataFrame, show_predictions: bool):
        for item in self._treeview.get_children():
            self._treeview.delete(item)

        if matches_df.shape[0] == 0:
            return

        fixture_day = self._day_var.get()

        if fixture_day < 10:
            fixture_day = f'0{fixture_day}'

        fixture_month = self._fixture_months[self._month_var.get()]
        fixture_year = self._matches_df.iloc[0]['Date'].split('/')[2]
        matches_df.insert(loc=0, column='Date', value=f'{fixture_day}/0{fixture_month}/{fixture_year}')
        matches_df['Home Team'] = matches_df['Home Team'].replace('', self._all_teams[0])
        matches_df['Away Team'] = matches_df['Away Team'].replace('', self._all_teams[1])
        matches_df = matches_df.replace('', '1.00')

        for i, values in enumerate(matches_df.values.tolist()):
            self._treeview.insert(parent='', index=i, values=values)

        if show_predictions:
            self._highlight_items(matches_df=matches_df)

    def _highlight_items(self, matches_df: pd.DataFrame):
        task = self._task_var.get()
        model_config = self._model_configs[task][self._model_id_var.get()]

        if task == 'Result':
            _, home_percent_prob = model_config.home_fixture_percentile
            _, draw_percent_prob = model_config.draw_fixture_percentile
            _, away_percent_prob = model_config.away_fixture_percentile

            home_ids = matches_df['Prob-H'].astype(float) >= home_percent_prob
            draw_ids = matches_df['Prob-D'].astype(float) >= draw_percent_prob
            away_ids = matches_df['Prob-A'].astype(float) >= away_percent_prob
            mask = (home_ids | draw_ids) | away_ids
        elif task == 'Over':
            _, under_percent_prob = model_config.under_fixture_percentile
            _, over_percent_prob = model_config.over_fixture_percentile

            under_ids = matches_df['Prob-U(2.5)'].astype(float) >= under_percent_prob
            over_ids = matches_df['Prob-O(2.5)'].astype(float) >= over_percent_prob
            mask = (under_ids | over_ids)
        else:
            raise NotImplementedError(f'Undefined task: "{task}"')

        if mask.sum() == 0:
            return

        previously_selected_items = self._treeview.selection()

        if len(previously_selected_items) > 0:
            self._treeview.selection_remove(previously_selected_items)

        items = self._treeview.get_children()
        selections = [item for item, is_selected in zip(items, mask) if is_selected]
        self._treeview.selection_set(selections)

    def _create_widgets(self):
        Label(self.window, text='Task:', font=('Arial', 11)).place(x=40, y=20)
        task_cb = Combobox(
            self.window, values=list(self._tasks.keys()), width=10, font=('Arial', 10), state='readonly', textvariable=self._task_var
        )
        task_cb.bind('<<ComboboxSelected>>', self._adjust_task)
        task_cb.place(x=100, y=20)

        Label(self.window, text='Model:', font=('Arial', 11)).place(x=220, y=20)
        self._model_cb = Combobox(
            self.window, width=14, font=('Arial', 10), state='readonly', textvariable=self._model_id_var
        )
        self._model_cb.place(x=290, y=20)

        Label(self.window, text='Browser:', font=('Arial', 11)).place(x=440, y=20)
        browser_cb = Combobox(
            self.window, width=10, values=config.browsers, font=('Arial', 10), state='readonly', textvariable=self._browser_var
        )
        browser_cb.current(0)
        browser_cb.place(x=520, y=20)

        Label(self.window, text='Date:', font=('Arial', 11)).place(x=640, y=20)
        month_cb = Combobox(
            self.window, width=8, values=config.months, font=('Arial', 10), state='readonly', textvariable=self._month_var
        )
        month_cb.current(0)
        month_cb.place(x=690, y=20)
        day_cb = Combobox(
            self.window, width=6, values=config.days, font=('Arial', 10), state='readonly', textvariable=self._day_var
        )
        day_cb.current(0)
        day_cb.place(x=790, y=20)

        self._parse_btn = Button(self.window, text='Load Fixtures', state='disabled', command=self._parse_fixture)
        self._parse_btn.place(x=40, y=610)
        self._predict_btn = Button(self.window, state='disabled', text='Predict Fixtures', command=self._predict)
        self._predict_btn.place(x=340, y=610)
        self._export_btn = Button(self.window, state='disabled', text='Export Fixtures', command=self._export)
        self._export_btn.place(x=640, y=610)

    def _parse_fixture(self):
        def parse_matches():
            result = FootyStatsScraper(browser=self._browser_var.get()).parse_matches(
                fixtures_url=self._league_config.league.fixtures_url,
                date_str=f'{self._month_var.get()} {self._day_var.get()}'
            )

            if isinstance(result, str):
                messagebox.showerror(parent=self.window, title='Parsing Error', message=result)
            else:
                matched_home_teams, matched_away_teams = match_fixture_teams(
                    parsed_home_teams=result['Home Team'].values.tolist(),
                    parsed_away_teams=result['Away Team'].values.tolist(),
                    unique_league_teams=set(self._all_teams).copy()
                )
                result['Home Team'] = pd.Series(matched_home_teams)
                result['Away Team'] = pd.Series(matched_away_teams)
                self._add_items(matches_df=result, show_predictions=False)
                self._predict_btn['state'] = 'normal'

                messagebox.showwarning(
                    parent=self.window,
                    title='Validate Fixture',
                    message='Validate the fixture matches and odds once parsing is completed!'
                            '\nDouble-click on incorrect items (teams or odds) to edit them before making predictions.'
                )

        TaskDialog(
            master=self.window,
            title='Parsing Fixtures',
            task=parse_matches,
            args=()
        ).start()

    def _predict(self):
        def predict_and_display(model) -> (np.ndarray, np.ndarray):
            tree_rows = [self._treeview.item(row)["values"][1:6] for row in self._treeview.get_children()]
            x = np.vstack([
                self._dataset_preprocessor.construct_input(
                    matches_df=self._matches_df,
                    home_team=row[0],
                    away_team=row[1],
                    odd_1=float(row[2]),
                    odd_x=float(row[3]),
                    odd_2=float(row[4])
                )
                for row in tree_rows
            ])
            y_prob = model.predict_proba(x)
            y_pred = y_prob.argmax(axis=1)

            task = self._task_var.get()
            for i in range(y_pred.shape[0]):
                tree_rows[i] += [self._task_predictions[task][y_pred[i]]] + [str(round(prob, 2)) for prob in y_prob[i]]

            matches_df = pd.DataFrame(data=tree_rows, columns=self._treeview_columns[task][1:])
            self._add_items(matches_df=matches_df, show_predictions=True)
            self._export_btn['state'] = 'normal'

        model_id = self._model_id_var.get()

        if model_id == '':
            messagebox.showerror(parent=self.window, title='Incorrect Configuration', message='Select mode to predict the fixtures')
            return

        model_config = self._model_configs[self._task_var.get()][self._model_id_var.get()]
        model = self._model_repository.load_model(model_config=model_config)

        TaskDialog(
            master=self.window,
            title='Predicting Fixture',
            task=predict_and_display,
            args=(model,)
        ).start()

    def _export(self):
        selected_items = self._treeview.selection()

        if len(selected_items) == 0:
            messagebox.showerror(parent=self.window, title='No Selected Items', message='Select matches predictions (CTRL + Click)')
            return

        fixture_filepath = filedialog.askopenfilename(
            defaultextension='.csv',
            filetypes=[("CSV files", "*.csv")]
        )

        data = [self._treeview.item(item, 'values') for item in selected_items]
        fixture_df = pd.DataFrame(data=data, columns=self._treeview_columns[self._task_var.get()])

        if fixture_filepath == '':
            fixture_filepath = filedialog.asksaveasfilename(
                defaultextension='.csv',
                filetypes=[("CSV files", "*.csv")]
            )
            fixture_df.to_csv(fixture_filepath, index=False, line_terminator='\n')
        else:
            df = pd.read_csv(fixture_filepath)

            if set(fixture_df.columns.tolist()) != set(df.columns.tolist()):
                messagebox.showerror(parent=self.window, title='Wrong CSV File', message='It looks like you selected a csv file from different task.')
                return

            fixture_df.to_csv(fixture_filepath, mode='a', header=False, index=False, line_terminator='\n')

        messagebox.showinfo('Exported', 'Done')


        # fixture_filepath = filedialog.asksaveasfilename(
        #     defaultextension='.csv',
        #     filetypes=[("CSV files", "*.csv")]
        # )
        #
        # if fixture_filepath is not None:
        #     try:
        #         df = pd.read_csv(fixture_filepath)
        #
        #         if set(self._treeview[self._task_var.get()]) != set(df.columns.tolist()):
        #             messagebox.showerror(parent=self.window, title='Wrong CSV File', message='It looks like you selected a csv file from different task.')
        #             return
        #         else:
        #             mode = 'a'
        #     except Exception as e:
        #         mode = 'w'
        #
        #     data = [self._treeview.item(item, 'values') for item in selected_items]
        #     fixture_df = pd.DataFrame(data=data, columns=self._treeview_columns[self._task_var.get()])
        #
        #     fixture_df.to_csv(fixture_filepath, mode=mode, index=False, line_terminator='\n')
        #     messagebox.showinfo('Exported', 'Done')

    def _init_dialog(self):
        messagebox.showinfo(
            parent=self.window,
            title='Fixtures Prediction',
            message='Load & predict the outcome of features. \nThen, the highlighted matches can be exported to a csv file.'
                    '\nIf there is existing csv file of prediction history, you can select it to append the new predictions, other choose CANCEL'
                    'and enter a new filename.'
        )

        webbrowser.open(url=self._league_config.league.fixtures_url)

    def _get_dialog_result(self):
        return None
