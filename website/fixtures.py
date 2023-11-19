import threading
import webbrowser
from datetime import datetime
from os import getcwd

import numpy as np
import pandas as pd
import requests
from flask_wtf import FlaskForm
from wtforms import SelectField

from database.repositories.model import ModelRepository
from fixtures.footystats.parser import FootyStatsFixtureParser
from models.ensemble import get_ensemble_predictions
from preprocessing.training import construct_inputs_from_fixtures


class FixturesForm(FlaskForm):
    model_name = SelectField("Select model")
    fixture_date = SelectField("Select fixture")
    def __init__(
        self,
        matches_df: pd.DataFrame,
        model_repository: ModelRepository,
        league_name: str,
        league_fixture_url: str,
    ):
        super().__init__()

        self._matches_df = matches_df
        self._model_repository = model_repository
        self._league_name = league_name
        self._league_fixture_url = league_fixture_url
        self._all_teams = set(matches_df["Home Team"].unique().tolist())

        self._treeview_columns = [
            "Date",
            "Home Team",
            "Away Team",
            "1",
            "X",
            "2",
            "Predicted",
            "Prob-H",
            "Prob-D",
            "Prob-A",
        ]
        self._fixture_months = {
            "Jan": "01",
            "Feb": "02",
            "Mar": "03",
            "Apr": "04",
            "May": "05",
            "Jun": "06",
            "Jul": "07",
            "Aug": "08",
            "Sep": "19",
            "Oct": "10",
            "Nov": "11",
            "Dec": "12",
        }
        self._saved_model_names = model_repository.get_all_models(
            league_name=league_name
        )
        self._fixture_parser = FootyStatsFixtureParser()


    def _add_items(
        self,
        items_df: pd.DataFrame,
        y_pred: np.ndarray or None,
        predict_proba: np.ndarray or None,
    ):
        if not "Date" in items_df:
            fixture_day = self._day_var.get()
            fixture_month = self._fixture_months[self._month_var.get()]
            fixture_year = self._matches_df.iloc[0]["Date"].split("/")[2]
            dates = [
                f"{fixture_day}/{fixture_month}/{fixture_year}"
                for _ in range(items_df.shape[0])
            ]
            items_df.insert(loc=0, column="Date", value=pd.Series(dates))

        if y_pred is None:
            y_out = np.array([" " for _ in range(items_df.shape[0])])
            for i, col in zip(
                [6, 7, 8, 9], ["Predicted", "Prob-H", "Prob-D", "Prob-A"]
            ):
                items_df.insert(loc=i, column=col, value=y_out)
        else:
            items_df["Predicted"] = y_pred
            for i, col in enumerate(["Prob-H", "Prob-D", "Prob-A"]):
                items_df[col] = predict_proba[:, i]
                items_df[col] = items_df[col].round(decimals=2)

            items_df["Prob-H"] = predict_proba[:, 0]
            items_df["Prob-D"] = predict_proba[:, 1]
            items_df["Prob-A"] = predict_proba[:, 2]

        items_df["Predicted"] = items_df["Predicted"].replace({0: "H", 1: "D", 2: "A"})
        for i, values in enumerate(items_df.values.tolist()):
            self._treeview.insert(parent="", index=i, values=values)

    def _download_fixture(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
        }
        response = requests.get(self._league_fixture_url, headers=headers)
        if response.status_code == 200:
            html_content = response.text
            current_time = datetime.now().strftime("%Y-%m-%d")
            filename = f"fixture_{current_time}.html"
            # Write the HTML content to the file
            self.fixture_path = f"{getcwd()}/database/storage/{filename}"
            with open(self.fixture_path, "w", encoding="utf-8") as file:
                file.write(html_content)
        else:
            print(f"Failed to download webpage. Status code: {response.status_code}")

    def _import_fixture(self) -> pd.DataFrame or str:
        self._download_fixture()
        webbrowser.open(self._league_fixture_url)

        if self.fixture_path is not None:
            parsing_result = self._fixture_parser.parse_fixture(
                fixture_filepath=self.fixture_path,
                fixtures_month=self._month_var.get(),
                fixtures_day=self._day_var.get(),
                unique_league_teams=self._all_teams,
            )

            if isinstance(parsing_result, pd.DataFrame):
                self._add_items(
                    items_df=parsing_result, y_pred=None, predict_proba=None
                )


    def _predict_fixture(self, fixture_df: pd.DataFrame):
        model_names = self._model_name_var.get()
        if model_names == "Ensemble":
            model_names = self._saved_model_names

        y_pred, predict_proba = self._predict(
            fixture_df=fixture_df, model_names=model_names
        )
        self._clear_items()
        self._add_items(items_df=fixture_df, y_pred=y_pred, predict_proba=predict_proba)
        self._export_predictions_btn["state"] = "normal"
        task_dialog.close()

    def _predict(
        self, fixture_df: pd.DataFrame, model_names: str or list
    ) -> (np.ndarray, np.ndarray):
        x = construct_inputs_from_fixtures(
            matches_df=self._matches_df, fixtures_df=fixture_df
        )

        if isinstance(model_names, str):
            model = self._model_repository.load_model(
                league_name=self._league_name,
                model_name=model_names,
                input_shape=x.shape[1:],
                random_seed=0,
            )
            y_pred, predict_proba = model.predict(x=x)
        else:
            models = [
                self._model_repository.load_model(
                    league_name=self._league_name,
                    model_name=name,
                    input_shape=x.shape[1:],
                    random_seed=0,
                )
                for name in model_names
            ]
            y_pred, predict_proba = get_ensemble_predictions(x=x, models=models)
        return y_pred, np.round(predict_proba, 2)

    def _submit_fixture_prediction(self):
        row_list = [
            self._treeview.item(row)["values"] for row in self._treeview.get_children()
        ]
        fixture_df = pd.DataFrame(data=row_list, columns=self._treeview_columns)

        if "" in fixture_df.values:
            messagebox.showerror(
                "Empty Cell",
                "Empty Cells have been found. Complete the missing values by double-clicking on empty cells",
            )
            return

        task_dialog = TaskDialog(self._window, self._title)
        task_thread = threading.Thread(
            target=self._predict_fixture, args=(task_dialog, fixture_df)
        )
        task_thread.start()
        task_dialog.open()

    def _export_predictions(self):
        fixture_filepath = filedialog.asksaveasfile(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        ).name

        if fixture_filepath is not None:
            row_list = [
                self._treeview.item(row)["values"]
                for row in self._treeview.get_children()
            ]
            fixture_df = pd.DataFrame(data=row_list, columns=self._treeview_columns)
            fixture_df.to_csv(fixture_filepath, index=False, line_terminator="\n")
            messagebox.showinfo("Exported", "Done")

