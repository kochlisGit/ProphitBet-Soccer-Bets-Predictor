import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDialog, QLabel, QPushButton, QFormLayout, QHBoxLayout
from src.analysis import DescriptiveAnalyzer
from src.gui.widgets.plot import PlotWindow


class DescriptiveAnalyzerWindow(QDialog):
    """ Descriptive statistics analyzer window using logistic regression. """

    def __init__(self, df: pd.DataFrame):
        super().__init__()

        self._analyzer = DescriptiveAnalyzer(df=df)

        self._seasons = self._analyzer.seasons
        self._feature_types = {'Home Team': 'home', 'Away Team': 'away'}
        self._colormap_dict = self._analyzer.colormap_dict

        self._title = 'Descriptive Analysis'
        self._width = 350
        self._height = 150

        # Declare UI placeholders.
        self._combo_season = None
        self._combo_feature = None
        self._combo_colormap = None

        self._initialize_window()
        self._add_widgets()

    def _initialize_window(self):
        self.setWindowTitle(self._title)
        self.resize(self._width, self._height)

        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )

    def _add_widgets(self):
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Season
        self._combo_season = QComboBox()
        self._combo_season.setFixedWidth(150)
        for season in self._seasons:
            self._combo_season.addItem(str(season))
        form.addRow(QLabel("Season:"), self._combo_season)

        # Team
        self._combo_feature = QComboBox()
        self._combo_feature.setFixedWidth(150)
        for team in self._feature_types:
            self._combo_feature.addItem(team)
        form.addRow(QLabel("Team:"), self._combo_feature)

        # Colormap
        self._combo_colormap = QComboBox()
        self._combo_colormap.setFixedWidth(150)
        for colormap in self._colormap_dict:
            self._combo_colormap.addItem(colormap)
        form.addRow(QLabel("Colormap:"), self._combo_colormap)

        # Analyze button (centered row)
        analyze_btn = QPushButton("Analyze")
        analyze_btn.setFixedWidth(150)
        analyze_btn.setFixedHeight(25)
        analyze_btn.clicked.connect(self._analyze)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(analyze_btn)
        btn_row.addStretch(1)
        form.addRow(btn_row)

        self.setLayout(form)

    def _analyze(self):
        season = self._seasons[self._combo_season.currentIndex()]
        feature_type = self._feature_types[self._combo_feature.currentText()]
        colormap = self._colormap_dict[self._combo_colormap.currentText()]
        ax = self._analyzer.generate_plot(season=season, colormap=colormap, feature_type=feature_type)
        PlotWindow(ax=ax, parent=self, title='Descriptive Statistics Analysis').show()
