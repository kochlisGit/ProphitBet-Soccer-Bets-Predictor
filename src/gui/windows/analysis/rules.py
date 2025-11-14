import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDialog, QLabel, QPushButton, QFormLayout, QHBoxLayout
from src.analysis import RuleExtractorAnalyzer
from src.gui.widgets.plot import PlotWindow
from src.preprocessing.utils.target import TargetType


class RulesAnalyzerWindow(QDialog):
    """ Rules-extractor window using Decision Tree. """

    def __init__(self, df: pd.DataFrame):
        super().__init__()

        self._analyzer = RuleExtractorAnalyzer(df=df)

        self._seasons = self._analyzer.seasons
        self._target_types = {'Result': TargetType.RESULT, 'O/U2.5': TargetType.OVER_UNDER}
        self._depths = [3, 4, 5, 6, 7]

        self._title = 'Rules Extractor Analysis'
        self._width = 350
        self._height = 150

        # Declare UI placeholders.
        self._combo_season = None
        self._combo_target = None
        self._combo_depth = None

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
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)  # optional

        # Season
        self._combo_season = QComboBox()
        self._combo_season.setFixedWidth(150)
        for season in self._seasons:
            self._combo_season.addItem(str(season))
        form.addRow(QLabel("Season:"), self._combo_season)

        # Target
        self._combo_target = QComboBox()
        self._combo_target.setFixedWidth(150)
        for target in self._target_types:
            self._combo_target.addItem(target)
        form.addRow(QLabel("Target:"), self._combo_target)

        # Max Depth
        self._combo_depth = QComboBox()
        self._combo_depth.setFixedWidth(150)
        for depth in self._depths:
            self._combo_depth.addItem(str(depth))
        form.addRow(QLabel("Max Depth:"), self._combo_depth)

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
        target_type = self._target_types[self._combo_target.currentText()]
        max_depth = self._depths[self._combo_target.currentIndex()]
        ax = self._analyzer.generate_plot(season=season, target_type=target_type, max_depth=max_depth)
        PlotWindow(ax=ax, parent=self, title='Tree-based Extracted Rules').show()
