import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDialog, QLabel, QPushButton, QFormLayout, QHBoxLayout
from src.analysis import DistributionAnalyzer
from src.gui.widgets.plot import PlotWindow


class DistributionAnalyzerWindow(QDialog):
    """ Distribution analyzer window using logistic regression. """

    def __init__(self, df: pd.DataFrame):
        super().__init__()

        self._analyzer = DistributionAnalyzer(df=df)

        self._seasons = self._analyzer.seasons
        self._columns = self._analyzer.all_features
        self._colormap_dict = self._analyzer.colormap_dict

        self._title = 'Distribution Analysis'
        self._width = 350
        self._height = 150

        # Declare UI placeholders.
        self._combo_season = None
        self._combo_column = None
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

        # Column
        self._combo_column = QComboBox()
        self._combo_column.setFixedWidth(150)
        for column in self._columns:
            self._combo_column.addItem(column)
        form.addRow(QLabel("Column:"), self._combo_column)

        # Colormap
        self._combo_colormap = QComboBox()
        self._combo_colormap.setFixedWidth(150)
        for colormap in self._colormap_dict:
            self._combo_colormap.addItem(colormap)
        form.addRow(QLabel("Colormap:"), self._combo_colormap)

        # Analyze button (centered)
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
        column = self._columns[self._combo_column.currentIndex()]
        colormap = self._colormap_dict[self._combo_colormap.currentText()]
        ax = self._analyzer.generate_plot(season=season, colormap=colormap, column=column)
        PlotWindow(ax=ax, parent=self, title='Distributions Plot').show()
