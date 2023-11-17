import os
import threading
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tkinter import StringVar, filedialog, messagebox
from tkinter.ttk import Treeview, Combobox, Label, Scrollbar, Button
from database.repositories.model import ModelRepository
from gui.dialogs.dialog import Dialog
from gui.dialogs.task import TaskDialog
from models.ensemble import get_ensemble_predictions
from preprocessing.training import preprocess_training_dataframe


class EvaluationDialog(Dialog):
    def __init__(
        self,
        root,
        matches_df: pd.DataFrame,
        model_repository: ModelRepository,
        league_name: str,
    ):
        super().__init__(
            root=root, title="Evaluation", window_size={"width": 900, "height": 700}
        )

        self._matches_df = matches_df
        self._model_repository = model_repository
        self._league_name = league_name
        self._saved_model_names = model_repository.get_all_models(
            league_name=league_name
        )

        self._num_eval_samples_var = StringVar(value="50")
        self._odd_filter_var = StringVar(value="None")
        self._selected_model_var = StringVar(value="None")
        self._acc_var = StringVar(value="N/A")
        self._f1_var = StringVar(value="N/A")
        self._prec_var = StringVar(value="N/A")
        self._rec_var = StringVar(value="N/A")

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

        self._treeview = None
        self._export_predictions_btn = None

    def _initialize(self):
        Label(self.window, text="Evaluation Samples:", font=("Arial", 12)).place(
            x=10, y=10
        )
        eval_samples_cb = Combobox(
            self.window,
            width=10,
            font=("Arial", 10),
            state="readonly",
            textvariable=self._num_eval_samples_var,
        )
        eval_samples_cb["values"] = ["25", "50", "100", "200"]
        eval_samples_cb.current(1)
        eval_samples_cb.place(x=170, y=10)

        Label(self.window, text="Odd Filter:", font=("Arial", 12)).place(x=300, y=10)
        odd_filter_cb = Combobox(
            self.window,
            width=15,
            font=("Arial", 10),
            state="readonly",
            textvariable=self._odd_filter_var,
        )
        odd_filter_cb["values"] = [
            "None",
            "1:(1.00 - 1.30)",
            "1:(1.31 - 1.60)",
            "1:(1.61 - 2.00)",
            "1:(2.01 - 3.00)",
            "1:>3.01",
            "X:(1.00 - 2.00)",
            "X:(2.01 - 3.00)",
            "X:(3.01 - 4.00)",
            "X:>4.00",
            "2:(1.00 - 1.30)",
            "2:(1.31 - 1.60)",
            "2:(1.61 - 2.00)",
            "2:(2.01 - 3.00)",
            "2:>3.01",
        ]
        odd_filter_cb.current(0)
        odd_filter_cb.place(x=390, y=10)

        model_names = ["None"] + self._saved_model_names
        if len(model_names) > 2:
            model_names.append("Ensemble")

        Label(self.window, text="Selected Model:", font=("Arial", 12)).place(
            x=580, y=10
        )
        selected_model_cb = Combobox(
            self.window,
            width=13,
            font=("Arial", 10),
            state="readonly",
            textvariable=self._selected_model_var,
        )
        selected_model_cb["values"] = model_names
        selected_model_cb.current(0)
        selected_model_cb.place(x=710, y=10)

        eval_samples_cb.bind("<<ComboboxSelected>>", self._submit_evaluation_task)
        odd_filter_cb.bind("<<ComboboxSelected>>", self._submit_evaluation_task)
        selected_model_cb.bind("<<ComboboxSelected>>", self._submit_evaluation_task)

        self._treeview = Treeview(
            self.window,
            columns=self._treeview_columns,
            show="headings",
            selectmode="extended",
            height=25,
        )
        for column_name in self._treeview_columns:
            self._treeview.column(column_name, anchor="center", stretch=True, width=70)
            self._treeview.heading(column_name, text=column_name, anchor="center")
        self._treeview.column("Home Team", anchor="center", stretch=True, width=100)
        self._treeview.column("Away Team", anchor="center", stretch=True, width=100)
        self._treeview.place(x=10, y=50)

        v_scroll = Scrollbar(
            self._window, orient="vertical", command=self._treeview.yview
        )
        v_scroll.place(x=870, y=50, height=550)
        self._treeview.configure(yscroll=v_scroll.set)

        Label(self.window, text="Accuracy:", font=("Arial", 12)).place(x=30, y=625)
        Label(
            self.window, font=("Arial", 10, "bold"), textvariable=self._acc_var
        ).place(x=110, y=626)

        Label(self.window, text="F1 Score:", font=("Arial", 12)).place(x=230, y=625)
        Label(self.window, font=("Arial", 10, "bold"), textvariable=self._f1_var).place(
            x=310, y=626
        )

        Label(self.window, text="Precision:", font=("Arial", 12)).place(x=535, y=625)
        Label(
            self.window, font=("Arial", 10, "bold"), textvariable=self._prec_var
        ).place(x=620, y=626)
        Label(self.window, text="Recall:", font=("Arial", 12)).place(x=535, y=645)
        Label(
            self.window, font=("Arial", 10, "bold"), textvariable=self._rec_var
        ).place(x=620, y=646)

        self._export_predictions_btn = Button(
            self.window,
            text="Export",
            state="disabled",
            command=self._export_predictions,
        )
        self._export_predictions_btn.place(x=390, y=650)

        self._add_items(
            matches_df=self._matches_df.iloc[0 : int(self._num_eval_samples_var.get())],
            y_pred=None,
            predict_proba=None,
        )

    def _clear_items(self):
        for item in self._treeview.get_children():
            self._treeview.delete(item)
        self._export_predictions_btn["state"] = "disabled"

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

            self._export_predictions_btn["state"] = "normal"

        items_df["Predicted"] = items_df["Predicted"].replace({0: "H", 1: "D", 2: "A"})
        for i, values in enumerate(items_df.values.tolist()):
            self._treeview.insert(parent="", index=i, values=values)

    def _highlight_filtered_samples(self, selected_ids: list or None):
        if selected_ids is None:
            previously_selected_items = self._treeview.selection()

            if len(previously_selected_items) > 0:
                self._treeview.selection_remove(self._treeview.selection())
            return

        items = self._treeview.get_children()
        selections = [items[index] for index in selected_ids]
        self._treeview.selection_set(selections)

    def _display_metrics(self, metrics: dict or None):
        if metrics is None:
            for var in [self._acc_var, self._f1_var, self._prec_var, self._rec_var]:
                var.set("N/A")
        else:
            for metric_name, score in metrics.items():
                if metric_name == "Accuracy":
                    accuracy, tp, n_samples = score
                    accuracy = np.round(accuracy * 100, 2)
                    self._acc_var.set(value=f"{accuracy}% - ({tp}/{n_samples})")
                else:
                    if metric_name == "F1-Score":
                        var = self._f1_var
                    elif metric_name == "Precision":
                        var = self._prec_var
                    elif metric_name == "Recall":
                        var = self._rec_var
                    else:
                        raise NotImplementedError(
                            f'Metric "{metric_name}" has not been implemented'
                        )

                    if len(score) == 3:
                        home_score, draw_score, away_score = np.round(score * 100, 2)
                        var.set(
                            value=f"H: {home_score}%, D: {draw_score}%, A {away_score}%"
                        )
                    else:
                        var.set("N/A")

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

        metrics = {
            "Accuracy": (
                accuracy_score(y_true=y, y_pred=y_pred),
                (y_pred == y).sum(),
                y_pred.shape[0],
            ),
            "F1-Score": f1_score(y_true=y, y_pred=y_pred, average=None),
            "Precision": precision_score(y_true=y, y_pred=y_pred, average=None),
            "Recall": recall_score(y_true=y, y_pred=y_pred, average=None),
        }
        return y_pred, np.round(predict_proba, 2), metrics

    def _evaluate_task(self, task_dialog: TaskDialog):
        num_eval_samples = int(self._num_eval_samples_var.get())
        odds_filter = self._odd_filter_var.get()
        model_name = self._selected_model_var.get()

        matches_df = self._matches_df.iloc[0:num_eval_samples]

        if odds_filter == "None":
            filtered_df = matches_df
        else:
            target_column, odd_range = odds_filter.split(":")
            if odd_range[0] == ">":
                left_range = float(odd_range[1:])
                right_range = 100.0
            else:
                ranges = odd_range[1:-1].replace(" ", "").split("-")
                left_range = float(ranges[0])
                right_range = float(ranges[1])
            filtered_ids = (matches_df[target_column] >= left_range) & (
                matches_df[target_column] <= right_range
            )
            filtered_df = matches_df[filtered_ids]

        if model_name != "None" and filtered_df.shape[0] > 0:
            if model_name == "Ensemble":
                model_name = self._saved_model_names
            y_pred, predict_proba, metrics = self._evaluate_model(
                matches_df=filtered_df, model_names=model_name
            )
        else:
            y_pred = predict_proba = None
            metrics = None

        self._clear_items()
        self._add_items(
            matches_df=filtered_df, y_pred=y_pred, predict_proba=predict_proba
        )
        self._display_metrics(metrics=metrics)
        task_dialog.close()

    def _submit_evaluation_task(self, event):
        task_dialog = TaskDialog(self._window, self._title)
        task_thread = threading.Thread(target=self._evaluate_task, args=(task_dialog,))
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

    def _dialog_result(self) -> None:
        return None
