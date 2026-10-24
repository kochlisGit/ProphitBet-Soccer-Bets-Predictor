from typing import Any, Dict, Optional
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSizePolicy, QVBoxLayout, QWidget
from src.core.database.league import LeagueDatabase
from src.gui.components.workspaces.page import WorkspacePage
from src.gui.widgets import IconLabel, PrimaryQPushButton


def _run_exploratory_data_analysis(df, selected_analyzers):
    """Import and run the heavy analysis engine inside the task process."""

    from src.core.analysis.eda import ExploratoryDataAnalyzer

    return ExploratoryDataAnalyzer().exploratory_data_analysis(
        df=df,
        eda_analyzers=selected_analyzers,
    )


class AnalysisOption(QPushButton):
    """ Analysis option card (similar to workspace card) but uses different style.
            * Icon
            * Title
            * Description

        Attributes
        ----------
        :param icon_filepath: (str) filepath of the icon.
        :param title: (str) title of the workspace.
        :param description: (str | None) description of the workspace.
        :param checked: (bool) whether to set initially as checked.
        :param parent: (QWidget | None) parent of the workspace.
    """

    def __init__(
            self,
            icon_filepath: str,
            title: str,
            description: str,
            checked: bool = True,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.setText('')

        # IMPORTANT: the card itself is deliberately NOT checkable. This prevents theme rules such as
        # QPushButton:checked from highlighting the whole row. Only the QCheckBox on the right owns checked state.
        self.setCheckable(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(82)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 14, 8)
        layout.setSpacing(0)

        icon_label = IconLabel(filepath=icon_filepath, size=62)
        layout.addWidget(icon_label)

        # Adding the title and description together as one compact block.
        text_wrapper = QWidget()
        text_wrapper.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        text_wrapper.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        text_layout = QVBoxLayout(text_wrapper)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)

        title_label = QLabel(title)
        title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        font = title_label.font()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        description_label = QLabel(description)
        description_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        font = description_label.font()
        font.setPointSize(11)
        description_label.setFont(font)
        description_label.setWordWrap(True)
        description_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        text_layout.addWidget(title_label)
        text_layout.addWidget(description_label)
        layout.addWidget(text_wrapper, stretch=1, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._checkbox = QCheckBox()
        self._checkbox.setCursor(Qt.CursorShape.PointingHandCursor)

        if checked:
            self.setChecked(True)

        layout.addWidget(self._checkbox, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Clicking anywhere on the row remains convenient, but the checkbox itself becomes checked/unchecked visually.
        self.clicked.connect(self._checkbox.toggle)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, checked: bool):
        self._checkbox.setChecked(checked)


class DataAnalysisWorkspace(WorkspacePage):
    """ Data Analysis workspace page. It has 7 analysis options:
            1. Dataset Analysis
            2. Univariate Analysis
            3. Bivariate Analysis
            4. Feature Selection
            5. Feature Importance
            6. Rules Extraction
            7. Team Performance Analysis

        Attributes
        ----------
        :param parent: (QWidget | None) parent of the workspace.
    """

    def __init__(
            self,
            settings: Dict[str, Any],
            database: LeagueDatabase,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(
            title='Data Analysis',
            description='Select the analyses you want to run.',
            parent=parent,
        )

        self._settings = settings
        self._database = database
        checked_analyzers = settings['analysis']['checked_analyzers']

        # Adding analysis options.
        self._options = {
            'Dataset': AnalysisOption(
                icon_filepath='storage/icons/menu/database.png',
                title='Dataset',
                description='Gain a high-level overview of the samples, the features and the dataset structure.',
                checked='Dataset' in checked_analyzers
            ),
            'Univariate': AnalysisOption(
                icon_filepath='storage/icons/menu/distribution.png',
                title='Univariate Analysis',
                description='Compute statistical properties of each feature and detect imbalances.',
                checked='Univariate' in checked_analyzers
            ),
            'Bivariate': AnalysisOption(
                icon_filepath='storage/icons/menu/regression.png',
                title='Bivariate Analysis',
                description='Calculate feature interactions, feature correlations and inspect pairwise interactions.',
                checked='Bivariate' in checked_analyzers
            ),
            'Selection': AnalysisOption(
                icon_filepath='storage/icons/menu/filter.png',
                title='Feature Selection',
                description='Investigate and rank features using qualitative information.',
                checked='Selection' in checked_analyzers
            ),
            'Importance': AnalysisOption(
                icon_filepath='storage/icons/menu/importance.png',
                title='Feature Importance',
                description='Analyze the importance of each feature with linear and non-linear methods.',
                checked='Importance' in checked_analyzers
            ),
            'Rules': AnalysisOption(
                icon_filepath='storage/icons/menu/rules.png',
                title='Rule Extraction',
                description='Employ a decision tree to derive classification rules.',
                checked='Rules' in checked_analyzers
            ),
            'Team': AnalysisOption(
                icon_filepath='storage/icons/menu/shield.png',
                title='Team Performance',
                description='Analyze the performance of each team by season.',
                checked='Team' in checked_analyzers
            )
        }
        for _, widget in self._options.items():
            self.main_layout.addWidget(widget)
        self.main_layout.addStretch(1)

        # Adding primary button.
        analyze_button = PrimaryQPushButton(text='Analyze')
        analyze_button.setIcon(QIcon('storage/icons/menu/barplot.png'))
        analyze_button.setIconSize(QSize(24, 24))
        analyze_button.setFixedHeight(48)
        font = analyze_button.font()
        font.setPointSize(14)
        font.setBold(True)
        analyze_button.setFont(font)
        analyze_button.clicked.connect(self._analyze)

        self.main_layout.addWidget(analyze_button)

    @property
    def options(self) -> Dict[str, AnalysisOption]:
        return self._options

    def _analyze(self):
        selected_analyzers = [name for name, option in self._options.items() if option.isChecked()]
        self._settings['analysis']['checked_analyzers'] = selected_analyzers

        if not selected_analyzers:
            QMessageBox.information(self, 'No Analysis Selected', 'Select at least one analysis option.')
            return

        df = getattr(self.window(), '_df', None)
        league_id = getattr(self.window(), 'current_league_id', None)
        if df is None or league_id is None:
            QMessageBox.warning(self, 'No League Loaded', 'Load a league before running data analysis.')
            return

        existing_report = self._database.get_analysis(league_id=league_id)
        if existing_report is not None:
            load_existing = QMessageBox.question(
                self,
                'Existing Analysis',
                f'An analysis already exists for "{league_id}". Do you want to load the existing analysis?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            ) == QMessageBox.StandardButton.Yes
            if load_existing:
                from src.gui.components.dialogs.analysis.report import AnalysisReport

                AnalysisReport(report_dict=existing_report, parent=self.window()).exec()
                return

        # The analysis engine imports SciPy, scikit-learn, Boruta, dcor, and
        # dtreeviz. Keep that import in the task process so the progress dialog
        # appears immediately instead of freezing the GUI before it opens.
        from src.gui.components.dialogs.tasks import Task

        report_dict = Task(
            _run_exploratory_data_analysis,
            df=df,
            selected_analyzers=selected_analyzers,
            title='Data Analysis',
            description='Applying the selected analyzers to the league data.',
            parent=self.window()
        ).run()

        if report_dict is None:
            return

        self._database.store_analysis(league_id=league_id, report_dict=report_dict)

        from src.gui.components.dialogs.analysis.report import AnalysisReport

        AnalysisReport(report_dict=report_dict, parent=self.window()).exec()
