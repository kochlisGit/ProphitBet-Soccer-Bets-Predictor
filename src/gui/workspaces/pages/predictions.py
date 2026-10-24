from typing import Any, Dict, Optional
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from src.gui.components.workspaces.page import WorkspacePage
from src.gui.widgets import IconLabel, PrimaryQPushButton


class PredictionCard(QPushButton):
    """ Workspace action card for Predictions workspace. Each card must have the following items:
            * Icon
            * Title
            * Description

        Attributes
        ----------
        :param icon_filepath: (str) filepath of the icon.
        :param title: (str) title of the workspace.
        :param description: (str | None) description of the workspace.
        :param parent: (QWidget | None) parent of the workspace.
    """

    def __init__(
            self,
            icon_filepath: str,
            title: str,
            description: str,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.setText('')
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(170)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 14, 0)
        layout.setSpacing(0)

        layout.addWidget(IconLabel(icon_filepath, 100))

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(3)

        title_label = QLabel(title)
        font = title_label.font()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        description_label = QLabel(description)
        font = description_label.font()
        font.setPointSize(11)
        description_label.setFont(font)
        description_label.setWordWrap(True)
        description_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        text_layout.addStretch()
        text_layout.addWidget(title_label)
        text_layout.addWidget(description_label)
        text_layout.addStretch()
        layout.addLayout(text_layout, 1)


class PredictionsWorkspace(WorkspacePage):
    """ Workspace Page for fixture Predictions. It includes 2 options:
            1. Offline fixture prediction.
            2. Online fixture prediction.

        Attributes
        ----------
        :param parent: (QWidget | None) parent widget.
    """

    def __init__(self, settings: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(
            title='Predictions',
            description='Enter/Download fixtures and predict the match outcomes using the trained models.',
            parent=parent
        )

        self._settings = settings

        # Adding the fixture Hero.
        hero = QFrame()
        hero.setFrameShape(QFrame.Shape.StyledPanel)

        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(0, 18, 20, 18)
        hero_layout.setSpacing(0)
        hero_layout.addWidget(IconLabel('storage/icons/menu/fixtures.png', 205))

        hero_text = QVBoxLayout()
        hero_text.setContentsMargins(0, 0, 0, 0)
        hero_text.setSpacing(0)
        hero_text.addStretch(1)

        # The arrow occupies its own column. Title and description both use column 1,
        # so their left edges are exactly aligned.
        hero_content = QGridLayout()
        hero_content.setContentsMargins(0, 0, 0, 0)
        hero_content.setHorizontalSpacing(6)
        hero_content.setVerticalSpacing(3)

        hero_content.addWidget(
            IconLabel('arrow.png', 38),
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )

        title_label = QLabel('Predict Fixture')
        font = title_label.font()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        hero_content.addWidget(title_label, 0, 1)

        description_label = QLabel(
            'Manually enter a fixture or automatically download upcoming '
            'fixtures and generate probabilistic predictions and betting strategies using your '
            'trained models and agents.'
        )
        font = description_label.font()
        font.setPointSize(12)
        description_label.setFont(font)
        description_label.setWordWrap(True)
        hero_content.addWidget(description_label, 1, 1)
        hero_content.setColumnStretch(1, 1)

        hero_text.addLayout(hero_content)
        hero_text.addStretch(1)
        hero_layout.addLayout(hero_text, 1)
        self.main_layout.addWidget(hero)

        # Adding predictions action cards.
        modes_layout = QHBoxLayout()
        modes_layout.setSpacing(10)

        self.manual_mode = PredictionCard(
            icon_filepath='storage/icons/menu/fixtures-offline.png',
            title='Manual Entry',
            description='Enter the teams, date, competition, and fixture information manually.'
        )
        self.online_mode = PredictionCard(
            icon_filepath='storage/icons/menu/fixtures-online.png',
            title='Download Fixtures',
            description='Automatically download upcoming fixtures.'
        )

        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.manual_mode)
        group.addButton(self.online_mode)
        self.manual_mode.setChecked(True)

        modes_layout.addWidget(self.manual_mode)
        modes_layout.addWidget(self.online_mode)
        self.main_layout.addLayout(modes_layout)
        self.main_layout.addStretch(1)

        predict_button = PrimaryQPushButton(text='Start Predictions')
        predict_button.setIcon(QIcon('storage/icons/menu/goal.png'))
        predict_button.setIconSize(QSize(24, 24))
        predict_button.setFixedHeight(48)
        font = predict_button.font()
        font.setPointSize(14)
        font.setBold(True)
        predict_button.setFont(font)
        predict_button.clicked.connect(self._predict)

        self.main_layout.addWidget(predict_button)

    def _predict(self):
        main_window = self.window()
        df = getattr(main_window, '_df', None)
        league_config = getattr(main_window, '_league_config', None)
        if df is None or df.empty:
            QMessageBox.warning(
                main_window,
                'No League Loaded',
                'Load a league before creating predictions.',
            )
            return

        model_database = getattr(main_window, 'model_database', None)
        agent_database = getattr(main_window, 'agent_database', None)
        if model_database is None or agent_database is None:
            QMessageBox.warning(
                main_window,
                'No League Loaded',
                'Load a league before creating predictions.',
            )
            return

        # Keep prediction/model dependencies behind the user action so this
        # feature does not add work to Main Menu initialization.
        if self.online_mode.isChecked():
            fixture_url = None if league_config is None else league_config.league.fixture_url
            if not fixture_url:
                QMessageBox.warning(
                    main_window,
                    'Fixture URL Missing',
                    'The active league does not define a FootyStats fixture URL.',
                )
                return

            from src.gui.components.dialogs.predictions.fixture import FixturePredictionDialog

            dialog = FixturePredictionDialog(
                df=df,
                fixture_url=fixture_url,
                model_database=model_database,
                agent_database=agent_database,
                parent=main_window,
            )
        else:
            from src.gui.components.dialogs.predictions.manual import ManualPredictionDialog

            dialog = ManualPredictionDialog(
                df=df,
                model_database=model_database,
                agent_database=agent_database,
                parent=main_window,
            )
        dialog.exec()
        dialog.deleteLater()
