import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDialog, QLabel, QPushButton, QFormLayout, QHBoxLayout
from src.analysis import VarianceAnalyzer
from src.gui.widgets.plot import PlotWindow


class VarianceAnalyzerWindow(QDialog):
    """ Variance analyzer window. """

    def __init__(self, df: pd.DataFrame):
        super().__init__()

        self._analyzer = VarianceAnalyzer(df=df)

        self._seasons = self._analyzer.seasons
        self._feature_types = {'Home Team': 'home', 'Away Team': 'away'}
        self._colormap_dict = self._analyzer.colormap_dict

        self._title = 'Variance Analysis'
        self._width = 350
        self._height = 120

        # Declare UI placeholders.
        self._combo_season = None
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

        # Colormap
        self._combo_colormap = QComboBox()
        self._combo_colormap.setFixedWidth(150)
        for colormap in self._colormap_dict:
            self._combo_colormap.addItem(colormap)
        form.addRow(QLabel("Colormap:"), self._combo_colormap)

        # Analyze button (centered in its row)
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
        colormap = self._colormap_dict[self._combo_colormap.currentText()]
        ax = self._analyzer.generate_plot(season=season, colormap=colormap)
        PlotWindow(ax=ax, parent=self, title='Variance Analysis').show()
