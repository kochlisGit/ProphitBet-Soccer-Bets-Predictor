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
        self._download_fixture()

        self.model_name.choices = self._saved_model_names
        self.fixture_date.choices = self._fixture_parser.get_available_match_tables(self.fixture_path)



    def _add_items(
        self,
        items_df: pd.DataFrame,
        y_pred: np.ndarray or None,
        predict_proba: np.ndarray or None,
    ):
        if not "Date" in items_df:
            fixture_day = self.fixture_date.data.split(" ")[1]
            fixture_month = self._fixture_months[ self.fixture_date.data.split(" ")[0]]
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
        return items_df

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

    def import_fixture(self) -> pd.DataFrame or str:
        if self.fixture_path is not None:
            parsing_result = self._fixture_parser.parse_fixture(
                fixture_filepath=self.fixture_path,
                fixture_date=self.fixture_date.data,
                unique_league_teams=self._all_teams,
            )
        return parsing_result


    def predict_fixture(self, fixture_df: pd.DataFrame):
        model_names = self.model_name.data
        if model_names == "Ensemble":
            model_names = self._saved_model_names

        y_pred, predict_proba = self._predict(
            fixture_df=fixture_df, model_names=model_names
        )
        return self._add_items(items_df=fixture_df, y_pred=y_pred, predict_proba=predict_proba)

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