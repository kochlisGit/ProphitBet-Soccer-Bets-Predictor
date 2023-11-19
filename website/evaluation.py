import numpy as np
import pandas as pd
from flask_wtf import FlaskForm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from wtforms import SelectField

from database.repositories.model import ModelRepository
from models.ensemble import get_ensemble_predictions
from preprocessing.training import preprocess_training_dataframe


class EvaluationForm(FlaskForm):
    n_evaluation_samples = SelectField("Nº evaluation samples")
    model = SelectField("Model")
    def __init__(
        self,
        model_repository: ModelRepository,
        league_name: str,
        matches_df: pd.DataFrame,
    ):
        super().__init__()

        self._matches_df = matches_df
        self._model_repository = model_repository
        self._league_name = league_name
        self._saved_model_names = model_repository.get_all_models(
            league_name=league_name
        )
        self.n_evaluation_samples.choices = [str(i) for i in range(10, 301, 50)]
        self.model.choices = self._saved_model_names


        self._treeview_columns = [
            "Index",
            "Home Team",
            "Away Team",
            "1",
            "X",
            "2",
            "Result",
            "Predicted",
            "Prob-H",
            "Prob-D",
            "Prob-A",
        ]
        self._acc_var = None
        self._f1_var = None
        self._prec_var = None
        self._rec_var = None


    def _add_items(
        self,
        matches_df: pd.DataFrame,
        y_pred: np.ndarray or None,
        predict_proba: np.ndarray or None,
    ):
        items_df = matches_df[["Home Team", "Away Team", "1", "X", "2", "Result"]]
        items_df.insert(
            loc=0, column="Index", value=np.arange(1, items_df.shape[0] + 1)
        )

        if y_pred is None:
            y_out = np.array([" " for _ in range(items_df.shape[0])])
            for i, col in zip(
                [7, 8, 9, 10], ["Predicted", "Prob-H", "Prob-D", "Prob-A"]
            ):
                items_df.insert(loc=i, column=col, value=y_out)
        else:
            items_df.insert(loc=7, column="Predicted", value=y_pred)
            for i, col in enumerate(["Prob-H", "Prob-D", "Prob-A"]):
                items_df.insert(loc=8 + i, column=col, value=predict_proba[:, i])
                items_df[col] = items_df[col].round(decimals=2)
        items_df["Predicted"].replace({0: "H", 1: "D", 2: "A"}, inplace=True)
        return items_df


    def _evaluate_model(
        self, matches_df: pd.DataFrame, model_names: str or list
    ) -> (np.ndarray, np.ndarray, dict):
        x, y = preprocess_training_dataframe(matches_df=matches_df, one_hot=False)

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
        def create_print_dict(res):
            names = ["Home", "Draw", "Away"]
            return {k: v for k, v in zip(names, res)}

        metrics = {
            "Accuracy": (
                accuracy_score(y_true=y, y_pred=y_pred),
                (y_pred == y).sum(),
                y_pred.shape[0],
            )[0],
            "F1-Score": create_print_dict(f1_score(y_true=y, y_pred=y_pred, average=None)),
            "Precision": create_print_dict(precision_score(y_true=y, y_pred=y_pred, average=None)),
            "Recall": create_print_dict(recall_score(y_true=y, y_pred=y_pred, average=None)),
        }
        return y_pred, np.round(predict_proba, 2), metrics

    def submit_evaluation_task(self):
        num_eval_samples = int(self.n_evaluation_samples.data)
        model_name = self.model.data

        matches_df = self._matches_df.iloc[0:num_eval_samples].copy(deep=True)

        filtered_df = matches_df

        if model_name != "None" and filtered_df.shape[0] > 0:
            if model_name == "Ensemble":
                model_name = self._saved_model_names
            y_pred, predict_proba, metrics = self._evaluate_model(
                matches_df=filtered_df, model_names=model_name
            )
        else:
            y_pred = predict_proba = None
            metrics = None
        items_df = self._add_items(
            matches_df=filtered_df, y_pred=y_pred, predict_proba=predict_proba
        )
        return items_df, metrics

    def _export_predictions(self):
        # fixture_filepath = filedialog.asksaveasfile(
        #     defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        # ).name

        # if fixture_filepath is not None:
        #     row_list = [
        #         self._treeview.item(row)["values"]
        #         for row in self._treeview.get_children()
        #     ]
        #     fixture_df = pd.DataFrame(data=row_list, columns=self._treeview_columns)
        #     fixture_df.to_csv(fixture_filepath, index=False, line_terminator="\n")
        #     messagebox.showinfo("Exported", "Done")
        pass
