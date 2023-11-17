from analysis.features.correlation import CorrelationAnalyzer
from analysis.features.classes import ClassDistributionAnalyzer
from analysis.features.importance import ImportanceAnalyzer
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64
import io
from wtforms import SelectField, BooleanField
from wtforms.validators import InputRequired
from flask_wtf import FlaskForm


class Plotter:
    def generate_image(self):
        fig, ax = plt.subplots(figsize=(15, 15))
        ax = self.do_plot(ax)
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        output.seek(0)
        encoded_image = base64.b64encode(output.read()).decode("utf-8")
        return encoded_image

    def do_plot(self):
        raise NotImplemented


class CorrelationPlotter(Plotter, FlaskForm):
    selected_color = SelectField("Color", validators=[InputRequired()])
    selected_columns = SelectField("Columns", validators=[InputRequired()])
    top_triangle = BooleanField("Show top triangle")

    def __init__(self, matches_df, *args, **kwargs) -> None:
        super(CorrelationPlotter, self).__init__(*args, **kwargs)
        self.matches_df = matches_df
        self._analyzer = CorrelationAnalyzer(matches_df=self.matches_df)
        self.selected_color.choices = [
            color_map.value for color_map in self._analyzer.ColorMaps
        ]
        self.selected_columns.choices = ["Home Columns", "Away Columns", "All"]

    def do_plot(self, ax: Axes) -> Axes:
        columns = (
            self._analyzer.home_columns
            if self.selected_columns.data == "Home Columns"
            else self._analyzer.away_columns
            if self.selected_columns.data == "Away Columns"
            else self._analyzer.away_columns + self._analyzer.home_columns
        )
        return self._analyzer.plot(
            columns=columns,
            ax=ax,
            color_map=self.selected_color.data,
            hide_upper_triangle=not self.top_triangle.data,
        )


class ClassDistributionPlotter(Plotter, FlaskForm):
    def __init__(self, matches_df, *args, **kwargs) -> None:
        super(ClassDistributionPlotter, self).__init__(*args, **kwargs)
        self.matches_df = matches_df
        self._analyzer = ClassDistributionAnalyzer(matches_df=self.matches_df)

    def do_plot(self, ax: Axes) -> Axes:
        return self._analyzer.plot(ax=ax)


class ImportancePlotter(Plotter, FlaskForm):
    selected_method = SelectField("Method", validators=[InputRequired()])

    def __init__(self, matches_df, *args, **kwargs) -> None:
        super(ImportancePlotter, self).__init__(*args, **kwargs)
        self.matches_df = matches_df
        self._analyzer = ImportanceAnalyzer(matches_df=self.matches_df)

        self._methods = {
            "Variance Analysis": self._analyzer.plot_feature_variances,
            "Univariate Test Importance": self._analyzer.plot_univariate_test_importance,
            "Classifier Importance Weights": self._analyzer.plot_feature_classification_weights,
            "Feature Elimination Importance": self._analyzer.plot_feature_elimination_importance,
        }
        self.selected_method.choices = list(self._methods.keys())

    def do_plot(self, ax: Axes) -> Axes:
        method_name = self.selected_method.data
        return self._methods[method_name](ax=ax)
