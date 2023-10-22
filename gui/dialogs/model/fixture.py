import threading
import numpy as np
import pandas as pd
import webbrowser
from tkinter import messagebox, filedialog, StringVar
from tkinter.ttk import Treeview, Combobox, Label, Scrollbar, Button, Entry
from database.repositories.model import ModelRepository
from fixtures.footystats.parser import FootyStatsFixtureParser
from gui.dialogs.dialog import Dialog
from gui.dialogs.task import TaskDialog
from gui.widgets.utils import validate_float_positive_entry
from models.ensemble import get_ensemble_predictions
from preprocessing.training import construct_inputs_from_fixtures


class FixturesDialog(Dialog):
    def __init__(
            self,
            root,
            matches_df: pd.DataFrame,
            model_repository: ModelRepository,
            league_name: str,
            league_fixture_url: str
    ):
        super().__init__(root=root, title='Fixture Prediction', window_size={'width': 900, 'height': 690})

        self._matches_df = matches_df
        self._model_repository = model_repository
        self._league_name = league_name
        self._league_fixture_url = league_fixture_url
        self._all_teams = set(matches_df['Home Team'].unique().tolist())

        self._treeview_columns = [
            'Date', 'Home Team', 'Away Team', '1', 'X', '2', 'Predicted', 'Prob-H', 'Prob-D', 'Prob-A'
        ]
        self._fixture_months = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
            'Jul': '07', 'Aug': '08', 'Sep': '19', 'Oct': '10', 'Nov': '11', 'Dec': '12',
        }
        self._saved_model_names = model_repository.get_all_models(league_name=league_name)
        self._fixture_parser = FootyStatsFixtureParser()

        self._model_name_var = StringVar()
        self._month_var = StringVar()
        self._day_var = StringVar()

        self._treeview = None
        self._export_predictions_btn = None
        self._predict_btn = None

    def _initialize(self):
        Button(self.window, text='Import Fixture', command=self._import_fixture).place(x=20, y=20)

        Label(self.window, font=('Arial', 12), text='Fixture Date:').place(x=260, y=20)

        months_cb = Combobox(self.window, state='readonly', width=10, textvariable=self._month_var)
        months_cb['values'] = list(self._fixture_months.keys())
        months_cb.current(0)
        months_cb.place(x=370, y=20)

        days = [str(i) for i in range(1, 32)]
        days_cb = Combobox(self.window, state='readonly', width=10, textvariable=self._day_var)
        days_cb['values'] = days
        days_cb.current(0)
        days_cb.place(x=485, y=20)

        self._export_predictions_btn = Button(
            self.window, text='Export Predictions', state='disabled', command=self._export_predictions
        )
        self._export_predictions_btn.place(x=680, y=20)

        self._treeview = Treeview(
            self.window,
            columns=self._treeview_columns,
            show='headings',
            selectmode='browse',
            height=25
        )
        for column_name in self._treeview_columns:
            self._treeview.column(column_name, anchor='center', stretch=True, width=70)
            self._treeview.heading(column_name, text=column_name, anchor='center')
        self._treeview.column('Date', anchor='center', stretch=True, width=100)
        self._treeview.column('Home Team', anchor='center', stretch=True, width=100)
        self._treeview.column('Away Team', anchor='center', stretch=True, width=100)
        self._treeview.place(x=10, y=60)
        self._treeview.bind('<Double-1>', self._edit_item_on_double_click_event)

        v_scroll = Scrollbar(self._window, orient='vertical', command=self._treeview.yview)
        v_scroll.place(x=850, y=50, height=550)
        self._treeview.configure(yscroll=v_scroll.set)

        model_names = self._saved_model_names if len(self._saved_model_names) == 1 \
            else self._saved_model_names + ['Ensemble']
        Label(self.window, font=('Arial', 12), text='Select Model:').place(x=200, y=630)
        model_cb = Combobox(self.window, state='readonly', width=15, textvariable=self._model_name_var)
        model_cb['values'] = model_names
        model_cb.current(0)
        model_cb.place(x=320, y=630)

        self._predict_btn = Button(
            self.window, text='Predict Fixture', state='disabled', command=self._submit_fixture_prediction
        )
        self._predict_btn.place(x=550, y=630)

        messagebox.showinfo(
            'Fixture Parsing',
            'Click on "Import Fixture to load "Fixture\'s URL into your web-browser and save the HTML page locally.'
            'The page can be saved using Control+S (CTRL+S) or right click and then "Save Us". '
            'Then, specify its filepath to parse the fixture. Before you proceed'
            'make sure you set the correct date of the fixture you are trying to parse'
        )

    def _clear_items(self):
        for item in self._treeview.get_children():
            self._treeview.delete(item)

    def _add_items(self, items_df: pd.DataFrame, y_pred: np.ndarray or None, predict_proba: np.ndarray or None):
        if not 'Date' in items_df:
            fixture_day = self._day_var.get()
            fixture_month = self._fixture_months[self._month_var.get()]
            fixture_year = self._matches_df.iloc[0]['Date'].split('/')[2]
            dates = [f'{fixture_day}/{fixture_month}/{fixture_year}' for _ in range(items_df.shape[0])]
            items_df.insert(loc=0, column='Date', value=pd.Series(dates))

        if y_pred is None:
            y_out = np.array([' ' for _ in range(items_df.shape[0])])
            for i, col in zip([6, 7, 8, 9], ['Predicted', 'Prob-H', 'Prob-D', 'Prob-A']):
                items_df.insert(loc=i, column=col, value=y_out)
        else:
            items_df['Predicted'] = y_pred
            for i, col in enumerate(['Prob-H', 'Prob-D', 'Prob-A']):
                items_df[col] = predict_proba[:, i]
                items_df[col] = items_df[col].round(decimals=2)

            items_df['Prob-H'] = predict_proba[:, 0]
            items_df['Prob-D'] = predict_proba[:, 1]
            items_df['Prob-A'] = predict_proba[:, 2]

        items_df['Predicted'] = items_df['Predicted'].replace({0: 'H', 1: 'D', 2: 'A'})
        for i, values in enumerate(items_df.values.tolist()):
            self._treeview.insert(parent='', index=i, values=values)

    def _edit_item_on_double_click_event(self, event):
        item = self._treeview.selection()[0]
        column_index = int(self._treeview.identify_column(event.x).replace('#', '')) - 1
        column_name = self._treeview_columns[column_index]
        current_value = self._treeview.item(item, 'values')[column_index]

        if column_name == 'Home Team' or column_name == 'Away Team':
            team_var = StringVar(value=current_value)
            cb = Combobox(self.window, state='readonly', width=15, textvariable=team_var)
            cb['values'] = sorted(self._all_teams)
            cb.place(x=50, y=50)

            def on_cb_selected(event):
                self._treeview.set(item, column=column_name, value=team_var.get())
                cb.destroy()
            cb.bind("<<ComboboxSelected>>", on_cb_selected)
        else:
            if column_name != 'Date':
                validate_float = self.window.register(validate_float_positive_entry)
                entry = Entry(self.window, validate='key', validatecommand=(validate_float, '%P'))
            else:
                entry = Entry(self.window)
            entry.insert(0, current_value)
            entry.place(x=50, y=50)

            def on_entry_return(event):
                self._treeview.set(item, column=column_name, value=entry.get())
                entry.destroy()
            entry.bind("<Return>", on_entry_return)

    def _import_fixture(self) -> pd.DataFrame or str:
        webbrowser.open(self._league_fixture_url)
        fixture_filepath = filedialog.askopenfilename(filetypes=[("HTML files", "*.html")])

        if fixture_filepath is not None:
            parsing_result = self._fixture_parser.parse_fixture(
                fixture_filepath=fixture_filepath,
                fixtures_month=self._month_var.get(),
                fixtures_day=self._day_var.get(),
                unique_league_teams=self._all_teams
            )

            if isinstance(parsing_result, pd.DataFrame):
                messagebox.showinfo(
                    'Parsing Result: OK',
                    'Matches have been imported. You can edit any cell by double-clicking on it')
                self._clear_items()
                self._add_items(items_df=parsing_result, y_pred=None, predict_proba=None)
                self._predict_btn['state'] = 'normal'
            else:
                messagebox.showerror('Parsing Result: ERROR', parsing_result)

    def _predict_fixture(self, task_dialog: TaskDialog, fixture_df: pd.DataFrame):
        model_names = self._model_name_var.get()
        if model_names == 'Ensemble':
            model_names = self._saved_model_names

        y_pred, predict_proba = self._predict(fixture_df=fixture_df, model_names=model_names)
        self._clear_items()
        self._add_items(items_df=fixture_df, y_pred=y_pred, predict_proba=predict_proba)
        self._export_predictions_btn['state'] = 'normal'
        task_dialog.close()

    def _predict(self, fixture_df: pd.DataFrame, model_names: str or list) -> (np.ndarray, np.ndarray):
        x = construct_inputs_from_fixtures(matches_df=self._matches_df, fixtures_df=fixture_df)

        if isinstance(model_names, str):
            model = self._model_repository.load_model(
                league_name=self._league_name, model_name=model_names, input_shape=x.shape[1:], random_seed=0
            )
            y_pred, predict_proba = model.predict(x=x)
        else:
            models = [
                self._model_repository.load_model(
                    league_name=self._league_name, model_name=name, input_shape=x.shape[1:], random_seed=0
                )
                for name in model_names
            ]
            y_pred, predict_proba = get_ensemble_predictions(x=x, models=models)
        return y_pred, np.round(predict_proba, 2)

    def _submit_fixture_prediction(self):
        row_list = [self._treeview.item(row)["values"] for row in self._treeview.get_children()]
        fixture_df = pd.DataFrame(data=row_list, columns=self._treeview_columns)

        if '' in fixture_df.values:
            messagebox.showerror(
                'Empty Cell',
                'Empty Cells have been found. Complete the missing values by double-clicking on empty cells')
            return

        task_dialog = TaskDialog(self._window, self._title)
        task_thread = threading.Thread(target=self._predict_fixture, args=(task_dialog, fixture_df))
        task_thread.start()
        task_dialog.open()

    def _export_predictions(self):
        fixture_filepath = filedialog.asksaveasfile(
            defaultextension='.csv',
            filetypes=[("CSV files", "*.csv")]
        ).name

        if fixture_filepath is not None:
            row_list = [self._treeview.item(row)["values"] for row in self._treeview.get_children()]
            fixture_df = pd.DataFrame(data=row_list, columns=self._treeview_columns)
            fixture_df.to_csv(fixture_filepath, index=False, line_terminator='\n')
            messagebox.showinfo('Exported', 'Done')

    def _dialog_result(self) -> None:
        return None
